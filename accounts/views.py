from __future__ import annotations

import logging

from adrf.views import APIView
from asgiref.sync import sync_to_async
from django.contrib.auth import aauthenticate, alogin, alogout, get_user_model
from django.middleware.csrf import get_token
from rest_framework import permissions, status
from rest_framework.response import Response

from accounts.serializers import AuthUserSerializer

logger = logging.getLogger(__name__)

User = get_user_model()


class RegisterAPIView(APIView):
    permission_classes = [permissions.AllowAny]

    async def post(self, request):
        username = (request.data.get("username") or "").strip()
        password = request.data.get("password") or ""
        password_confirm = request.data.get("password_confirm") or ""

        if not username or not password:
            return Response(
                {"detail": "Username and password are required."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if len(username) < 3:
            return Response(
                {"detail": "Username must be at least 3 characters."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if len(password) < 8:
            return Response(
                {"detail": "Password must be at least 8 characters."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if password != password_confirm:
            return Response(
                {"detail": "Passwords do not match."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        exists = await sync_to_async(User.objects.filter(username=username).exists)()
        if exists:
            return Response(
                {"detail": "Username already taken."},
                status=status.HTTP_409_CONFLICT,
            )

        user = await sync_to_async(User.objects.create_user)(username=username, password=password)
        await alogin(request, user)

        return Response({
            "success": True,
            "user": AuthUserSerializer(user).data,
            "csrf_token": get_token(request),
        }, status=status.HTTP_201_CREATED)


class LoginAPIView(APIView):
    permission_classes = [permissions.AllowAny]

    async def post(self, request):
        username = (request.data.get("username") or "").strip()
        password = request.data.get("password") or ""

        if not username or not password:
            return Response(
                {"detail": "Username and password are required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        user = await aauthenticate(request, username=username, password=password)
        if user is None:
            return Response(
                {"detail": "Invalid credentials."},
                status=status.HTTP_401_UNAUTHORIZED,
            )

        await alogin(request, user)
        return Response({
            "success": True,
            "user": AuthUserSerializer(user).data,
            "csrf_token": get_token(request),
        })


class LogoutAPIView(APIView):
    async def post(self, request):
        await alogout(request)
        return Response({"success": True, "csrf_token": get_token(request)})


class SessionAPIView(APIView):
    permission_classes = [permissions.AllowAny]

    async def get(self, request):
        if request.user.is_authenticated:
            data = AuthUserSerializer(request.user).data
            return Response({
                "authenticated": True,
                "user": data,
                "csrf_token": get_token(request),
            })
        return Response({"authenticated": False, "csrf_token": get_token(request)})
