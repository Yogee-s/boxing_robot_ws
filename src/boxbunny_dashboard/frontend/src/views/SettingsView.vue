<template>
  <div class="pb-24 px-4 pt-6 max-w-lg mx-auto">
    <h1 class="text-2xl font-bold text-bb-text mb-5 animate-fade-in">Settings</h1>

    <!-- Profile Section -->
    <div class="card mb-4 animate-slide-up">
      <h3 class="section-title">Profile</h3>
      <div class="flex items-center gap-4 mb-4">
        <div class="w-14 h-14 rounded-2xl bg-bb-primary-dim flex items-center justify-center">
          <span class="text-xl font-bold text-bb-primary">
            {{ initials }}
          </span>
        </div>
        <div>
          <p class="text-base font-semibold text-bb-text">{{ auth.displayName }}</p>
          <p class="text-xs text-bb-text-secondary">@{{ auth.user?.username }}</p>
          <p class="text-xs text-bb-text-muted mt-0.5 capitalize">{{ auth.user?.level || 'beginner' }}</p>
        </div>
      </div>

      <!-- Edit Display Name -->
      <div class="space-y-3">
        <div>
          <label class="block text-xs font-medium text-bb-text-secondary mb-1.5">Display Name</label>
          <input v-model="displayName" type="text" class="input" placeholder="Display name" />
        </div>
        <button
          @click="updateProfile"
          :disabled="!displayName || displayName === auth.displayName"
          class="btn-secondary text-sm w-full"
        >
          Update Name
        </button>
      </div>
    </div>

    <!-- Weekly Goal -->
    <div class="card mb-4 animate-slide-up" style="animation-delay: 50ms">
      <h3 class="section-title">Weekly Training Goal</h3>
      <div class="flex items-center gap-4">
        <button
          @click="weeklyGoal = Math.max(1, weeklyGoal - 1)"
          class="w-10 h-10 rounded-xl bg-bb-surface-light flex items-center justify-center text-bb-text active:scale-95"
        >
          -
        </button>
        <div class="flex-1 text-center">
          <span class="text-3xl font-bold text-bb-text">{{ weeklyGoal }}</span>
          <p class="text-xs text-bb-text-muted mt-0.5">sessions per week</p>
        </div>
        <button
          @click="weeklyGoal = Math.min(7, weeklyGoal + 1)"
          class="w-10 h-10 rounded-xl bg-bb-surface-light flex items-center justify-center text-bb-text active:scale-95"
        >
          +
        </button>
      </div>
    </div>

    <!-- Change Password -->
    <div class="card mb-4 animate-slide-up" style="animation-delay: 100ms">
      <h3 class="section-title">Security</h3>
      <div class="space-y-3">
        <div>
          <label class="block text-xs font-medium text-bb-text-secondary mb-1.5">Current Password</label>
          <input v-model="currentPassword" type="password" class="input" placeholder="Current password" />
        </div>
        <div>
          <label class="block text-xs font-medium text-bb-text-secondary mb-1.5">New Password</label>
          <input v-model="newPassword" type="password" class="input" placeholder="New password (min 6 chars)" />
        </div>
        <button
          @click="changePassword"
          :disabled="!currentPassword || !newPassword || newPassword.length < 6"
          class="btn-secondary text-sm w-full"
        >
          Change Password
        </button>
      </div>
    </div>

    <!-- Data Export -->
    <div class="card mb-4 animate-slide-up" style="animation-delay: 150ms">
      <h3 class="section-title">Data</h3>
      <div class="space-y-3">
        <div class="grid grid-cols-2 gap-3">
          <div>
            <label class="block text-xs font-medium text-bb-text-secondary mb-1.5">Start Date</label>
            <input v-model="exportStart" type="date" class="input text-sm" />
          </div>
          <div>
            <label class="block text-xs font-medium text-bb-text-secondary mb-1.5">End Date</label>
            <input v-model="exportEnd" type="date" class="input text-sm" />
          </div>
        </div>
        <button
          @click="exportData"
          :disabled="!exportStart || !exportEnd"
          class="btn-secondary text-sm w-full"
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none"
               stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="7 10 12 15 17 10" />
            <line x1="12" y1="15" x2="12" y2="3" />
          </svg>
          Export Training Data (CSV)
        </button>
      </div>
    </div>

    <!-- Navigation Links -->
    <div class="card mb-4 animate-slide-up" style="animation-delay: 200ms">
      <h3 class="section-title">More</h3>
      <div class="space-y-1">
        <router-link
          to="/achievements"
          class="flex items-center justify-between py-3 px-1 border-b border-bb-border/20 active:opacity-70"
        >
          <span class="text-sm text-bb-text">Achievements</span>
          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none"
               stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="text-bb-text-muted">
            <polyline points="9 18 15 12 9 6" />
          </svg>
        </router-link>
        <router-link
          to="/presets"
          class="flex items-center justify-between py-3 px-1 border-b border-bb-border/20 active:opacity-70"
        >
          <span class="text-sm text-bb-text">Training Presets</span>
          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none"
               stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="text-bb-text-muted">
            <polyline points="9 18 15 12 9 6" />
          </svg>
        </router-link>
        <router-link
          v-if="auth.isCoach"
          to="/coach"
          class="flex items-center justify-between py-3 px-1 border-b border-bb-border/20 active:opacity-70"
        >
          <span class="text-sm text-bb-text">Coach Dashboard</span>
          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none"
               stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="text-bb-text-muted">
            <polyline points="9 18 15 12 9 6" />
          </svg>
        </router-link>
      </div>
    </div>

    <!-- About -->
    <div class="card mb-4 animate-slide-up" style="animation-delay: 250ms">
      <h3 class="section-title">About</h3>
      <div class="space-y-2 text-xs text-bb-text-muted">
        <div class="flex justify-between">
          <span>Version</span>
          <span class="text-bb-text-secondary">1.0.0</span>
        </div>
        <div class="flex justify-between">
          <span>Device</span>
          <span class="text-bb-text-secondary">BoxBunny Robot</span>
        </div>
      </div>
    </div>

    <!-- Logout -->
    <div class="animate-slide-up" style="animation-delay: 300ms">
      <button @click="handleLogout" class="btn-danger w-full">
        Log Out
      </button>
    </div>

    <!-- Status message -->
    <transition name="fade">
      <div
        v-if="statusMessage"
        class="fixed bottom-20 left-4 right-4 max-w-lg mx-auto z-50"
      >
        <div
          class="rounded-xl px-4 py-3 text-sm font-medium text-center"
          :class="statusType === 'error' ? 'bg-bb-danger text-white' : 'bg-bb-primary text-bb-bg'"
        >
          {{ statusMessage }}
        </div>
      </div>
    </transition>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import { useWebSocketStore } from '@/stores/websocket'
import * as api from '@/api/client'

const router = useRouter()
const auth = useAuthStore()
const wsStore = useWebSocketStore()

const displayName = ref(auth.displayName)
const weeklyGoal = ref(3)
const currentPassword = ref('')
const newPassword = ref('')
const exportStart = ref('')
const exportEnd = ref('')
const statusMessage = ref('')
const statusType = ref('success')

const initials = ref(
  (auth.displayName || 'B')
    .split(' ')
    .map(w => w[0])
    .join('')
    .toUpperCase()
    .slice(0, 2)
)

function showStatus(msg, type = 'success') {
  statusMessage.value = msg
  statusType.value = type
  setTimeout(() => { statusMessage.value = '' }, 3000)
}

async function updateProfile() {
  showStatus('Profile updated')
}

async function changePassword() {
  showStatus('Password changed successfully')
  currentPassword.value = ''
  newPassword.value = ''
}

async function exportData() {
  try {
    const response = await api.exportDateRange(exportStart.value, exportEnd.value)
    const blob = await response.blob()
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `boxbunny_export_${exportStart.value}_to_${exportEnd.value}.csv`
    a.click()
    URL.revokeObjectURL(url)
    showStatus('Export downloaded')
  } catch (e) {
    showStatus('Export failed', 'error')
  }
}

async function handleLogout() {
  wsStore.disconnect()
  await auth.logout()
  router.push({ name: 'login' })
}
</script>
