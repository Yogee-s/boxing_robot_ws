<template>
  <div class="pb-24 px-4 pt-6 max-w-lg mx-auto">
    <h1 class="text-2xl font-bold text-bb-text mb-5 animate-fade-in">Coach Dashboard</h1>

    <!-- Connection Status -->
    <div class="card mb-4 animate-slide-up">
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-3">
          <div
            class="w-3 h-3 rounded-full"
            :class="wsStore.connected ? 'bg-bb-primary animate-pulse' : 'bg-bb-danger'"
          />
          <span class="text-sm text-bb-text">
            {{ wsStore.connected ? 'Connected' : 'Disconnected' }}
          </span>
        </div>
        <span class="text-xs text-bb-text-muted">
          {{ participants.length }} participant{{ participants.length !== 1 ? 's' : '' }}
        </span>
      </div>
    </div>

    <!-- Active Station Controls -->
    <div class="card mb-4 animate-slide-up" style="animation-delay: 50ms">
      <h3 class="section-title">Station Control</h3>

      <div v-if="!activeSession">
        <div class="space-y-3">
          <div>
            <label class="block text-xs font-medium text-bb-text-secondary mb-1.5">Session Name</label>
            <input v-model="sessionName" type="text" class="input" placeholder="e.g., Morning Session" />
          </div>

          <!-- Preset selector -->
          <div v-if="presets.length > 0">
            <label class="block text-xs font-medium text-bb-text-secondary mb-1.5">Load Preset</label>
            <select v-model="selectedPresetId" class="input">
              <option :value="null">No preset</option>
              <option v-for="p in presets" :key="p.id" :value="p.id">
                {{ p.name }} ({{ p.preset_type }})
              </option>
            </select>
          </div>

          <button
            @click="startSession"
            :disabled="!sessionName"
            class="btn-primary w-full"
          >
            Start Station
          </button>
        </div>
      </div>

      <div v-else>
        <div class="flex items-center justify-between mb-4">
          <div>
            <p class="text-sm font-semibold text-bb-text">{{ activeSession.name }}</p>
            <p class="text-xs text-bb-text-muted">Session running</p>
          </div>
          <div class="badge-green animate-pulse-glow">LIVE</div>
        </div>
        <button @click="endSession" class="btn-danger w-full">
          End Session
        </button>
      </div>
    </div>

    <!-- Live Participants -->
    <div class="card mb-4 animate-slide-up" style="animation-delay: 100ms">
      <h3 class="section-title">Participants</h3>

      <div v-if="participantsLoading" class="space-y-3">
        <div v-for="i in 3" :key="i" class="skeleton h-14 w-full rounded-xl" />
      </div>

      <div v-else-if="participants.length === 0" class="py-8 text-center">
        <p class="text-bb-text-muted text-sm">No participants connected</p>
        <p class="text-bb-text-muted text-xs mt-1">Participants will appear when they join</p>
      </div>

      <div v-else class="space-y-2">
        <div
          v-for="p in participants"
          :key="p.username"
          class="flex items-center justify-between py-3 px-3 rounded-xl bg-bb-surface-light"
        >
          <div class="flex items-center gap-3">
            <div
              class="w-2 h-2 rounded-full"
              :class="p.connected ? 'bg-bb-primary' : 'bg-bb-text-muted'"
            />
            <div>
              <p class="text-sm font-medium text-bb-text">{{ p.display_name }}</p>
              <p class="text-[10px] text-bb-text-muted">@{{ p.username }}</p>
            </div>
          </div>
          <div class="text-right">
            <p class="text-sm font-bold text-bb-text">{{ p.score }}</p>
            <p class="text-[10px] text-bb-text-muted">{{ p.rounds_completed }} rounds</p>
          </div>
        </div>
      </div>
    </div>

    <!-- Past Coaching Sessions -->
    <div class="card animate-slide-up" style="animation-delay: 150ms">
      <h3 class="section-title">Past Sessions</h3>

      <div v-if="pastSessions.length === 0" class="py-6 text-center">
        <p class="text-bb-text-muted text-sm">No past coaching sessions</p>
      </div>

      <div v-else class="space-y-2">
        <div
          v-for="s in pastSessions"
          :key="s.session_id"
          class="flex items-center justify-between py-2 border-b border-bb-border/20 last:border-0"
        >
          <div>
            <p class="text-sm font-medium text-bb-text">{{ s.name }}</p>
            <p class="text-xs text-bb-text-muted">{{ s.started_at }}</p>
          </div>
          <span class="text-xs text-bb-text-secondary">
            {{ s.participant_count }} users
          </span>
        </div>
      </div>
    </div>

    <!-- Status toast -->
    <transition name="fade">
      <div
        v-if="statusMessage"
        class="fixed bottom-20 left-4 right-4 max-w-lg mx-auto z-50"
      >
        <div class="rounded-xl px-4 py-3 text-sm font-medium text-center bg-bb-primary text-bb-bg">
          {{ statusMessage }}
        </div>
      </div>
    </transition>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import { useWebSocketStore } from '@/stores/websocket'
import { useAuthStore } from '@/stores/auth'
import * as api from '@/api/client'

const wsStore = useWebSocketStore()
const auth = useAuthStore()

const sessionName = ref('')
const selectedPresetId = ref(null)
const activeSession = ref(null)
const participants = ref([])
const participantsLoading = ref(false)
const presets = ref([])
const pastSessions = ref([])
const statusMessage = ref('')

let pollInterval = null
let unsubscribe = null

function showStatus(msg) {
  statusMessage.value = msg
  setTimeout(() => { statusMessage.value = '' }, 3000)
}

async function loadPresets() {
  try {
    presets.value = await api.getPresets()
  } catch (e) {
    console.error('Failed to load presets:', e)
  }
}

async function fetchParticipants() {
  try {
    participants.value = await api.getLiveParticipants()
  } catch (e) {
    // Silently handle polling errors
  }
}

async function fetchPastSessions() {
  try {
    pastSessions.value = await api.getCoachingSessions()
  } catch (e) {
    console.error('Failed to fetch coaching sessions:', e)
  }
}

async function startSession() {
  if (!sessionName.value) return
  try {
    if (selectedPresetId.value) {
      await api.loadCoachConfig(selectedPresetId.value)
    }
    const result = await api.startStation(sessionName.value, selectedPresetId.value)
    activeSession.value = {
      id: result.coaching_session_id,
      name: sessionName.value,
    }
    showStatus('Station started')
    sessionName.value = ''
  } catch (e) {
    console.error('Failed to start station:', e)
  }
}

async function endSession() {
  if (!activeSession.value) return
  try {
    await api.endCoachSession(activeSession.value.id)
    showStatus('Session ended')
    activeSession.value = null
    await fetchPastSessions()
  } catch (e) {
    console.error('Failed to end session:', e)
  }
}

onMounted(async () => {
  // Connect WebSocket as coach
  if (auth.user) {
    wsStore.connect(auth.user.username, 'coach')
  }

  await Promise.all([
    loadPresets(),
    fetchParticipants(),
    fetchPastSessions(),
  ])

  // Poll for live participants
  pollInterval = setInterval(fetchParticipants, 5000)

  // Listen for WebSocket events
  unsubscribe = wsStore.on('session_stats', (data) => {
    const idx = participants.value.findIndex(p => p.username === data.username)
    if (idx >= 0) {
      participants.value[idx] = { ...participants.value[idx], ...data }
    }
  })
})

onUnmounted(() => {
  if (pollInterval) clearInterval(pollInterval)
  if (unsubscribe) unsubscribe()
})
</script>
