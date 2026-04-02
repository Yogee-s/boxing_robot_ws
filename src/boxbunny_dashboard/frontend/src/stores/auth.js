import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import * as api from '@/api/client'

export const useAuthStore = defineStore('auth', () => {
  const user = ref(null)
  const token = ref(api.getToken())
  const loading = ref(false)
  const error = ref(null)

  const isAuthenticated = computed(() => !!token.value)
  const displayName = computed(() => user.value?.display_name || 'Boxer')
  const isCoach = computed(() => user.value?.user_type === 'coach')

  async function initialize() {
    if (!token.value) return
    loading.value = true
    error.value = null
    try {
      const session = await api.getSession()
      user.value = session
    } catch (e) {
      // Token expired or invalid
      token.value = null
      user.value = null
      api.clearToken()
    } finally {
      loading.value = false
    }
  }

  async function login(username, password) {
    loading.value = true
    error.value = null
    try {
      const data = await api.login(username, password)
      token.value = data.token
      user.value = {
        user_id: data.user_id,
        username: data.username,
        display_name: data.display_name,
        user_type: data.user_type,
        level: 'beginner',
      }
      return data
    } catch (e) {
      error.value = e.message || 'Login failed'
      throw e
    } finally {
      loading.value = false
    }
  }

  async function signup(username, password, displayName, level) {
    loading.value = true
    error.value = null
    try {
      const data = await api.signup(username, password, displayName, level)
      token.value = data.token
      user.value = {
        user_id: data.user_id,
        username: data.username,
        display_name: data.display_name,
        user_type: data.user_type,
        level,
      }
      return data
    } catch (e) {
      error.value = e.message || 'Signup failed'
      throw e
    } finally {
      loading.value = false
    }
  }

  async function logout() {
    try {
      // Tell GUI to log out too
      await api.sendRemoteCommand('navigate', { route: 'auth' }).catch(() => {})
      await api.logout()
    } finally {
      token.value = null
      user.value = null
    }
  }

  // Restore session on store creation
  initialize()

  return {
    user,
    token,
    loading,
    error,
    isAuthenticated,
    displayName,
    isCoach,
    initialize,
    login,
    signup,
    logout,
  }
})
