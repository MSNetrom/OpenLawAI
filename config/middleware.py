from __future__ import annotations

from asgiref.sync import iscoroutinefunction, markcoroutinefunction, sync_to_async
from django.db import connections
from django.http import StreamingHttpResponse
from whitenoise.middleware import WhiteNoiseMiddleware


class AsyncWhiteNoiseMiddleware(WhiteNoiseMiddleware):
    """WhiteNoise with async-iterator responses to avoid ASGI sync-iterator warnings."""

    @staticmethod
    def serve(static_file, request):
        response = static_file.get_response(request.method, request.META)
        status = int(response.status)

        file_obj = response.file
        if file_obj:
            data = file_obj.read()
            file_obj.close()
        else:
            data = b""

        async def _aiter():
            yield data

        http_response = StreamingHttpResponse(_aiter(), status=status)
        del http_response["content-type"]
        for key, value in response.headers:
            http_response[key] = value
        return http_response


class AsyncDBConnectionMiddleware:
    """Close stale DB connections on executor threads after each async request.

    Django's request_finished signal fires on the main async thread, missing
    connections created on ThreadPoolExecutor threads by async ORM calls.
    """

    async_capable = True
    sync_capable = True

    def __init__(self, get_response):
        self.get_response = get_response
        self.is_async = iscoroutinefunction(get_response)
        if self.is_async:
            markcoroutinefunction(self)

    def __call__(self, request):
        if self.is_async:
            return self._acall(request)
        return self.get_response(request)

    async def _acall(self, request):
        try:
            return await self.get_response(request)
        finally:
            await sync_to_async(connections.close_all, thread_sensitive=True)()
