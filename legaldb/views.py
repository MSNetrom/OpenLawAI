from django.shortcuts import render


def chat_ui(request):
    return render(request, "chat/index.html", {"app_mode": "chat"})


def dev_chat_ui(request):
    return render(request, "chat/index.html", {"app_mode": "dev"})
