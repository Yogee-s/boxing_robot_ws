<template>
  <div class="flex flex-col h-screen h-[100dvh] bg-bb-bg">
    <!-- Header -->
    <div class="flex items-center gap-3 px-4 py-3 bg-bb-surface border-b border-bb-border/30 safe-top">
      <button @click="$router.back()" class="text-bb-text-secondary active:opacity-70">
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none"
             stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <polyline points="15 18 9 12 15 6" />
        </svg>
      </button>
      <div class="w-8 h-8 rounded-full bg-bb-green-dim flex items-center justify-center">
        <span class="text-bb-green text-xs font-bold">AI</span>
      </div>
      <div class="flex-1">
        <p class="text-sm font-semibold text-bb-text">BoxBunny Coach</p>
        <p class="text-[10px] text-bb-green">Online</p>
      </div>
    </div>

    <!-- Messages -->
    <div
      ref="messagesContainer"
      class="flex-1 overflow-y-auto px-4 py-4 space-y-3"
    >
      <!-- Welcome message -->
      <div v-if="messages.length === 0 && !loading" class="text-center py-12">
        <div class="w-16 h-16 mx-auto mb-4 rounded-2xl bg-bb-green-dim flex items-center justify-center">
          <span class="text-2xl font-bold text-bb-green">AI</span>
        </div>
        <p class="text-bb-text font-semibold mb-1">BoxBunny AI Coach</p>
        <p class="text-bb-text-muted text-sm max-w-xs mx-auto">
          Ask me anything about boxing technique, your training progress, or get personalized workout tips.
        </p>
        <!-- Quick prompts -->
        <div class="flex flex-wrap gap-2 justify-center mt-6">
          <button
            v-for="prompt in quickPrompts"
            :key="prompt"
            @click="sendMessage(prompt)"
            class="px-3 py-2 rounded-xl bg-bb-surface border border-bb-border/30 text-xs text-bb-text-secondary
                   active:scale-95 transition-transform"
          >
            {{ prompt }}
          </button>
        </div>
      </div>

      <!-- Message bubbles -->
      <div
        v-for="(msg, idx) in messages"
        :key="idx"
        class="flex animate-fade-in"
        :class="msg.role === 'user' ? 'justify-end' : 'justify-start'"
      >
        <div
          class="max-w-[80%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed"
          :class="msg.role === 'user'
            ? 'bg-bb-green text-bb-bg rounded-br-md'
            : 'bg-bb-surface border border-bb-border/30 text-bb-text rounded-bl-md'"
        >
          {{ msg.content }}
        </div>
      </div>

      <!-- Loading dots -->
      <div v-if="sending" class="flex justify-start animate-fade-in">
        <div class="bg-bb-surface border border-bb-border/30 rounded-2xl rounded-bl-md px-4 py-3">
          <div class="flex gap-1.5">
            <span class="w-2 h-2 bg-bb-text-muted rounded-full animate-bounce" style="animation-delay: 0ms" />
            <span class="w-2 h-2 bg-bb-text-muted rounded-full animate-bounce" style="animation-delay: 150ms" />
            <span class="w-2 h-2 bg-bb-text-muted rounded-full animate-bounce" style="animation-delay: 300ms" />
          </div>
        </div>
      </div>
    </div>

    <!-- Input -->
    <div class="px-4 py-3 bg-bb-surface border-t border-bb-border/30 safe-bottom">
      <form @submit.prevent="sendMessage()" class="flex gap-2">
        <input
          v-model="input"
          type="text"
          placeholder="Ask your AI coach..."
          class="input flex-1 py-2.5 text-sm"
          :disabled="sending"
          maxlength="2000"
        />
        <button
          type="submit"
          :disabled="!input.trim() || sending"
          class="w-10 h-10 rounded-xl bg-bb-green text-bb-bg flex items-center justify-center
                 active:scale-95 transition-transform disabled:opacity-40 disabled:pointer-events-none flex-shrink-0"
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none"
               stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <line x1="22" y1="2" x2="11" y2="13" />
            <polygon points="22 2 15 22 11 13 2 9 22 2" />
          </svg>
        </button>
      </form>
    </div>
  </div>
</template>

<script setup>
import { ref, nextTick, onMounted } from 'vue'
import * as api from '@/api/client'

const messages = ref([])
const input = ref('')
const sending = ref(false)
const loading = ref(true)
const messagesContainer = ref(null)

const quickPrompts = [
  'How can I improve my jab?',
  'Analyze my recent sessions',
  'Give me a training plan',
  'Tips for better defense',
]

async function scrollToBottom() {
  await nextTick()
  if (messagesContainer.value) {
    messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
  }
}

async function sendMessage(text = null) {
  const messageText = text || input.value.trim()
  if (!messageText) return

  // Add user message
  messages.value.push({
    role: 'user',
    content: messageText,
  })
  input.value = ''
  sending.value = true
  await scrollToBottom()

  try {
    const response = await api.sendChatMessage(messageText)
    messages.value.push({
      role: 'assistant',
      content: response.reply,
      timestamp: response.timestamp,
    })
  } catch (e) {
    messages.value.push({
      role: 'assistant',
      content: 'Sorry, I could not process your request. Please try again.',
    })
  } finally {
    sending.value = false
    await scrollToBottom()
  }
}

onMounted(async () => {
  try {
    const history = await api.getChatHistory(50)
    messages.value = history
  } catch (e) {
    // Start fresh if history fetch fails
  } finally {
    loading.value = false
    await scrollToBottom()
  }
})
</script>
