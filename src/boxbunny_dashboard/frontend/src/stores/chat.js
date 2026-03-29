import { defineStore } from 'pinia'
import { ref } from 'vue'
import * as api from '@/api/client'

export const useChatStore = defineStore('chat', () => {
  const messages = ref([])
  const loaded = ref(false)
  const sending = ref(false)

  async function loadHistory() {
    if (loaded.value) return
    try {
      const history = await api.getChatHistory(50)
      messages.value = Array.isArray(history) ? history : []
    } catch (e) {
      // Start fresh if history fetch fails
    } finally {
      loaded.value = true
    }
  }

  async function sendMessage(text) {
    if (!text || sending.value) return null

    messages.value.push({
      role: 'user',
      content: text,
      timestamp: new Date().toISOString(),
    })

    sending.value = true

    try {
      const response = await api.sendChatMessage(text)
      const assistantMsg = {
        role: 'assistant',
        content: response.reply,
        timestamp: response.timestamp || new Date().toISOString(),
        suggestions: response.suggestions || null,
      }
      messages.value.push(assistantMsg)
      return assistantMsg
    } catch (e) {
      const errorMsg = {
        role: 'assistant',
        content: 'Sorry, I could not process your request. Please try again.',
        timestamp: new Date().toISOString(),
      }
      messages.value.push(errorMsg)
      return errorMsg
    } finally {
      sending.value = false
    }
  }

  function clearMessages() {
    messages.value = []
    loaded.value = false
  }

  return {
    messages,
    loaded,
    sending,
    loadHistory,
    sendMessage,
    clearMessages,
  }
})
