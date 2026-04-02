import { defineStore } from 'pinia'
import { ref } from 'vue'
import * as api from '@/api/client'

export const useChatStore = defineStore('chat', () => {
  const messages = ref([])
  const loaded = ref(false)
  const sending = ref(false)
  const streaming = ref(false)

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
      const fullText = response.reply || ''
      const suggestions = response.suggestions || null
      const actions = response.actions || null
      const timestamp = response.timestamp || new Date().toISOString()

      // Add empty assistant message, then stream words into it
      const assistantMsg = {
        role: 'assistant',
        content: '',
        timestamp,
        suggestions: null,
      }
      messages.value.push(assistantMsg)
      const msgIdx = messages.value.length - 1

      // Reveal word by word
      sending.value = false
      streaming.value = true
      const words = fullText.split(/(\s+)/)
      for (let i = 0; i < words.length; i++) {
        messages.value[msgIdx] = {
          ...messages.value[msgIdx],
          content: messages.value[msgIdx].content + words[i],
        }
        // Small delay per word for streaming effect
        if (i < words.length - 1) {
          await new Promise(r => setTimeout(r, 30))
        }
      }
      // Show suggestions and actions after streaming completes
      messages.value[msgIdx] = {
        ...messages.value[msgIdx],
        suggestions,
        actions,
      }
      streaming.value = false
      return messages.value[msgIdx]
    } catch (e) {
      const errorMsg = {
        role: 'assistant',
        content: 'Sorry, I could not process your request. Please try again.',
        timestamp: new Date().toISOString(),
      }
      messages.value.push(errorMsg)
      sending.value = false
      streaming.value = false
      return errorMsg
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
    streaming,
    loadHistory,
    sendMessage,
    clearMessages,
  }
})
