from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from .models import ChatSession, ChatMessage
from .rag import retrieve_context

# 🔹 CHAT LIST (Landing Page)
@login_required
def chat_list(request):
    # Order by updated_at so the most recent activity is at the top
    chats = ChatSession.objects.filter(user=request.user).order_by("-updated_at")
    
    # If there's at least one chat, we can redirect to the most recent one
    if chats.exists():
        return redirect("chat_detail", chat_id=chats.first().id)
    
    return render(request, "chat/chat_list.html", {"chats": chats})

# 🔹 CREATE NEW CHAT
@login_required
def create_chat(request):
    # Start with a generic title; we will update it once the first message is sent
    chat = ChatSession.objects.create(
        user=request.user,
        title="New Conversation"
    )
    return redirect("chat_detail", chat_id=chat.id)

# 🔹 CHAT DETAIL (The main interaction view)
@login_required
def chat_detail(request, chat_id):
    chat = get_object_or_404(ChatSession, id=chat_id, user=request.user)
    
    # Consistently order sidebar by the last update time
    chats = ChatSession.objects.filter(user=request.user).order_by("-updated_at")
    messages = chat.messages.order_by("created_at")

    if request.method == "POST":
        user_input = request.POST.get("message", "").strip()

        if user_input:
            # 1. Save User Message
            ChatMessage.objects.create(chat=chat, role="user", content=user_input)
            
            # 2. Update Chat Title if it's the very first message
            if messages.count() == 0:
                # Use first 35 characters as the title
                new_title = user_input[:35] + ("..." if len(user_input) > 35 else "")
                chat.title = new_title
            
            # 3. Trigger 'updated_at' so this chat moves to the top of the sidebar
            chat.save()

            # 4. Get AI Response from your RAG logic
            response = retrieve_context(user_input)
            ChatMessage.objects.create(chat=chat, role="assistant", content=response)

        return redirect("chat_detail", chat_id=chat.id)

    return render(request, "chat/chat_detail.html", {
        "chat": chat,
        "messages": messages,
        "chats": chats,
        "active_chat": chat
    })