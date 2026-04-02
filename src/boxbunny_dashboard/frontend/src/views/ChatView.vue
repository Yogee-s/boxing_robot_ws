<template>
  <div class="flex flex-col bg-bb-bg" style="height: 100svh; height: -webkit-fill-available">
    <!-- Header -->
    <div class="flex items-center gap-3 px-4 py-3 bg-bb-surface border-b border-bb-border/30 safe-top">
      <button @click="$router.back()" class="text-bb-text-secondary active:opacity-70">
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none"
             stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <polyline points="15 18 9 12 15 6" />
        </svg>
      </button>
      <div class="w-8 h-8 rounded-full bg-gradient-to-br from-bb-primary/40 to-bb-primary/10 flex items-center justify-center ring-2 ring-bb-primary/20">
        <span class="text-bb-primary text-xs font-bold">AI</span>
      </div>
      <div class="flex-1">
        <p class="text-sm font-semibold text-bb-text">BoxBunny Coach</p>
        <div class="flex items-center gap-1.5">
          <span class="w-1.5 h-1.5 rounded-full animate-pulse"
                :class="llmReady ? 'bg-green-400' : 'bg-bb-warning'" />
          <p class="text-[10px]" :class="llmReady ? 'text-green-400' : 'text-bb-warning'">
            {{ llmReady ? 'Ready' : llmChecking ? 'Connecting...' : 'Loading model...' }}
          </p>
        </div>
      </div>
      <button
        @click="showContext = !showContext"
        class="text-bb-text-muted active:opacity-70 p-1"
      >
        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none"
             stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>
        </svg>
      </button>
    </div>

    <!-- Context Banner -->
    <transition name="slide-down">
      <div v-if="showContext" class="px-4 py-2.5 bg-bb-surface-light border-b border-bb-border/20">
        <div class="flex items-center gap-2">
          <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none"
               stroke="#FF6B35" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
          </svg>
          <p class="text-[11px] text-bb-text-secondary leading-snug">
            AI Coach knows your skill level, recent sessions, training goals, and performance history to give personalized advice.
          </p>
        </div>
      </div>
    </transition>

    <!-- Quick Action Chips -->
    <div v-if="chatStore.messages.length === 0 || showChips" class="px-4 py-2 flex gap-2 overflow-x-auto no-scrollbar border-b border-bb-border/10">
      <button
        v-for="chip in quickChips"
        :key="chip.text"
        @click="handleSend(chip.text)"
        class="flex-shrink-0 inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full
               bg-bb-surface border border-bb-border/30 text-[11px] text-bb-text-secondary
               active:scale-95 active:bg-bb-surface-light transition-all duration-150
               whitespace-nowrap"
      >
        <span class="text-xs">{{ chip.icon }}</span>
        {{ chip.label }}
      </button>
    </div>

    <!-- Messages -->
    <div
      ref="messagesContainer"
      class="flex-1 overflow-y-auto px-4 py-4 space-y-4"
    >
      <!-- Welcome message -->
      <div v-if="chatStore.messages.length === 0 && chatStore.loaded" class="text-center py-8">
        <div class="w-16 h-16 mx-auto mb-4 rounded-2xl bg-gradient-to-br from-bb-primary/30 to-bb-primary/5 flex items-center justify-center ring-1 ring-bb-primary/20">
          <span class="text-2xl font-bold text-bb-primary">AI</span>
        </div>
        <p class="text-bb-text font-semibold mb-1">BoxBunny AI Coach</p>
        <p class="text-bb-text-muted text-sm max-w-xs mx-auto leading-relaxed">
          Ask me anything about boxing technique, your training progress, or get personalized workout tips.
        </p>
      </div>

      <!-- Message bubbles -->
      <div
        v-for="(msg, idx) in chatStore.messages"
        :key="idx"
        class="flex animate-fade-in"
        :class="msg.role === 'user' ? 'justify-end' : 'justify-start'"
      >
        <div class="max-w-[82%] flex flex-col" :class="msg.role === 'user' ? 'items-end' : 'items-start'">
          <!-- Avatar for assistant -->
          <div v-if="msg.role === 'assistant' && (idx === 0 || chatStore.messages[idx-1]?.role !== 'assistant')"
               class="flex items-center gap-1.5 mb-1">
            <div class="w-5 h-5 rounded-full bg-bb-primary-dim flex items-center justify-center">
              <span class="text-bb-primary text-[8px] font-bold">AI</span>
            </div>
            <span class="text-[10px] text-bb-text-muted font-medium">Coach</span>
          </div>

          <!-- Bubble -->
          <div
            class="rounded-2xl px-4 py-2.5 text-sm leading-relaxed"
            :class="msg.role === 'user'
              ? 'bg-bb-primary text-white rounded-br-md'
              : 'bg-bb-surface border border-bb-border/30 text-bb-text rounded-bl-md'"
          >
            <span class="whitespace-pre-wrap">{{ msg.content }}</span>
          </div>

          <!-- Timestamp -->
          <span class="text-[9px] text-bb-text-muted mt-1 px-1">
            {{ formatTimestamp(msg.timestamp) }}
          </span>

          <!-- Training action cards (from LLM suggestions) -->
          <div v-if="msg.actions && msg.actions.length > 0" class="space-y-2 mt-2">
            <button
              v-for="(action, ai) in msg.actions" :key="ai"
              @click="startAction(action)"
              class="w-full flex items-center gap-3 p-3 rounded-xl
                     bg-bb-primary-dim border border-bb-primary/30
                     active:scale-[0.97] transition-all"
            >
              <div class="w-9 h-9 rounded-lg bg-bb-primary/20 flex items-center justify-center flex-shrink-0">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="#FF6B35" stroke="none">
                  <polygon points="5 3 19 12 5 21 5 3" />
                </svg>
              </div>
              <div class="text-left flex-1 min-w-0">
                <p class="text-sm font-semibold text-bb-primary">{{ action.label }}</p>
                <p v-if="action.config.combo_seq" class="text-[10px] text-bb-text-muted">
                  Combo: {{ action.config.combo_seq }} · {{ action.config['Work Time'] || '' }}
                </p>
              </div>
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none"
                   stroke="#FF6B35" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="flex-shrink-0">
                <polyline points="9 18 15 12 9 6"/>
              </svg>
            </button>
          </div>

          <!-- Follow-up suggestions (after assistant messages) -->
          <div v-if="msg.role === 'assistant' && idx === chatStore.messages.length - 1 && !chatStore.sending && !chatStore.streaming && followUpSuggestions.length > 0"
               class="flex flex-wrap gap-1.5 mt-2">
            <button
              v-for="suggestion in followUpSuggestions"
              :key="suggestion"
              @click="handleSend(suggestion)"
              class="px-2.5 py-1 rounded-lg bg-bb-surface-light border border-bb-border/20
                     text-[10px] text-bb-text-secondary active:scale-95 transition-transform"
            >
              {{ suggestion }}
            </button>
          </div>
        </div>
      </div>

      <!-- Typing indicator -->
      <div v-if="chatStore.sending" class="flex justify-start animate-fade-in">
        <div class="max-w-[82%] flex flex-col items-start">
          <div class="flex items-center gap-1.5 mb-1">
            <div class="w-5 h-5 rounded-full bg-bb-primary-dim flex items-center justify-center">
              <span class="text-bb-primary text-[8px] font-bold">AI</span>
            </div>
            <span class="text-[10px] text-bb-text-muted font-medium">Coach is typing</span>
          </div>
          <div class="bg-bb-surface border border-bb-border/30 rounded-2xl rounded-bl-md px-4 py-3">
            <div class="flex gap-1">
              <span class="typing-dot" />
              <span class="typing-dot" style="animation-delay: 200ms" />
              <span class="typing-dot" style="animation-delay: 400ms" />
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Input -->
    <div class="px-4 pt-3 bg-bb-surface border-t border-bb-border/30" style="padding-bottom: calc(env(safe-area-inset-bottom, 12px) + 16px)">
      <form @submit.prevent="handleSend()" class="flex gap-2 items-end">
        <div class="flex-1 relative">
          <input
            v-model="input"
            type="text"
            :placeholder="!llmReady ? 'AI Coach loading... please wait' : chatStore.sending ? 'AI Coach is thinking...' : chatStore.streaming ? 'Generating response...' : 'Ask your AI coach...'"
            class="input py-2.5 text-sm pr-10"
            :disabled="!llmReady || chatStore.sending || chatStore.streaming"
            maxlength="2000"
            @keydown.enter.prevent="handleSend()"
          />
          <span v-if="input.length > 0" class="absolute right-3 bottom-2.5 text-[9px] text-bb-text-muted">
            {{ input.length }}/2000
          </span>
        </div>
        <button
          type="submit"
          :disabled="!llmReady || !input.trim() || chatStore.sending || chatStore.streaming"
          class="w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0
                 transition-all duration-200 active:scale-90
                 disabled:opacity-30 disabled:pointer-events-none"
          :class="input.trim() ? 'bg-bb-primary text-bb-bg shadow-sm shadow-bb-primary/30' : 'bg-bb-surface-light text-bb-text-muted'"
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
import { ref, nextTick, onMounted, onUpdated, computed, watch } from 'vue'
import { useChatStore } from '@/stores/chat'
import * as api from '@/api/client'

const chatStore = useChatStore()
const input = ref('')
const messagesContainer = ref(null)
const showContext = ref(false)
const showChips = ref(true)
const llmReady = ref(false)
const llmChecking = ref(true)

const quickChips = [
  { icon: 'A', label: 'Analyze my last session', text: 'Analyze my last training session and give me detailed feedback.' },
  { icon: 'D', label: 'Suggest a drill', text: 'Suggest a drill that would help me improve based on my recent performance.' },
  { icon: 'P', label: 'Create a training plan', text: 'Create a personalized weekly training plan for me.' },
  { icon: 'F', label: "How's my form?", text: 'Based on my training data, how is my overall form and what should I focus on?' },
]

const followUpSuggestions = computed(() => {
  if (chatStore.messages.length === 0) return []
  const lastMsg = chatStore.messages[chatStore.messages.length - 1]
  if (lastMsg.role !== 'assistant') return []

  // If the API returned suggestions, use those
  if (lastMsg.suggestions && Array.isArray(lastMsg.suggestions)) {
    return lastMsg.suggestions.slice(0, 3)
  }

  // Generate contextual follow-ups based on content
  const content = (lastMsg.content || '').toLowerCase()
  if (content.includes('drill') || content.includes('training')) {
    return ['How many rounds?', 'Make it harder', 'Show me alternatives']
  }
  if (content.includes('reaction') || content.includes('speed')) {
    return ['How do I improve?', 'Compare to my average', 'Set a goal']
  }
  if (content.includes('defense') || content.includes('block')) {
    return ['Defence drill tips', 'Show my defense stats', 'Common mistakes?']
  }
  return ['Tell me more', 'What else?', 'How do I improve?']
})

// Auto-scroll whenever DOM updates (catches streaming word-by-word)
onUpdated(() => {
  if (messagesContainer.value && (chatStore.sending || chatStore.streaming)) {
    messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
  }
})

async function scrollToBottom() {
  await nextTick()
  if (messagesContainer.value) {
    messagesContainer.value.scrollTo({
      top: messagesContainer.value.scrollHeight,
      behavior: 'smooth',
    })
  }
}

async function startAction(action) {
  // Navigate GUI to the config/setup page — user starts when ready
  const config = { ...action.config }
  if (action.type === 'power_test' || action.type === 'reaction_test' || action.type === 'stamina_test') {
    await api.sendRemoteCommand('navigate', { route: action.config.route || action.type })
  } else {
    // Navigate to training config page so user can review and start
    await api.sendRemoteCommand('setup_drill', {
      combo: { name: config.name, seq: config.combo_seq || '', id: null },
      difficulty: config.difficulty || 'Beginner',
      Rounds: config.Rounds || '2',
      'Work Time': config['Work Time'] || '60s',
      'Rest Time': '30s',
      Speed: config.Speed || 'Medium (2s)',
    })
  }
  chatStore.messages.push({
    role: 'assistant',
    content: `Drill loaded on the robot — press Start or centre pad when you're ready!`,
    timestamp: new Date().toISOString(),
  })
  await scrollToBottom()
}

async function handleSend(text = null) {
  const messageText = text || input.value.trim()
  if (!messageText) return

  input.value = ''
  showChips.value = false
  await scrollToBottom()

  await chatStore.sendMessage(messageText)
  await scrollToBottom()
}

function formatTimestamp(ts) {
  if (!ts) return ''
  try {
    const d = new Date(ts)
    const now = new Date()
    const diffMs = now - d
    const diffMin = Math.floor(diffMs / 60000)

    if (diffMin < 1) return 'now'
    if (diffMin < 60) return `${diffMin}m ago`

    const isToday = d.toDateString() === now.toDateString()
    if (isToday) {
      return d.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true })
    }

    const yesterday = new Date(now)
    yesterday.setDate(yesterday.getDate() - 1)
    if (d.toDateString() === yesterday.toDateString()) {
      return 'Yesterday ' + d.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true })
    }

    return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }) + ' ' +
           d.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true })
  } catch {
    return ''
  }
}

onMounted(async () => {
  await chatStore.loadHistory()
  await scrollToBottom()
  // Check if LLM is ready — poll every 3s until ready
  const checkLlm = async () => {
    try {
      const status = await api.getChatStatus()
      llmReady.value = status.ready
    } catch {
      llmReady.value = false
    }
    llmChecking.value = false
  }
  await checkLlm()
  if (!llmReady.value) {
    const interval = setInterval(async () => {
      await checkLlm()
      if (llmReady.value) clearInterval(interval)
    }, 3000)
  }
})
</script>

<style scoped>
.typing-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background-color: #616161;
  animation: typingBounce 1.4s ease-in-out infinite;
}

@keyframes typingBounce {
  0%, 60%, 100% {
    transform: translateY(0);
    opacity: 0.4;
  }
  30% {
    transform: translateY(-6px);
    opacity: 1;
  }
}

.no-scrollbar::-webkit-scrollbar {
  display: none;
}
.no-scrollbar {
  -ms-overflow-style: none;
  scrollbar-width: none;
}

.slide-down-enter-active {
  transition: all 0.25s ease-out;
}
.slide-down-leave-active {
  transition: all 0.2s ease-in;
}
.slide-down-enter-from {
  opacity: 0;
  transform: translateY(-10px);
  max-height: 0;
}
.slide-down-leave-to {
  opacity: 0;
  transform: translateY(-10px);
  max-height: 0;
}
</style>
