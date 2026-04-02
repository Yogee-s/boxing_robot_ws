<template>
  <div class="pb-24 px-4 pt-6 max-w-lg mx-auto">
    <h1 class="text-2xl font-bold text-bb-text mb-1 animate-fade-in">Training</h1>
    <p class="text-sm text-bb-text-muted mb-5 animate-fade-in">Start training directly on the robot from your phone</p>

    <!-- Quick Start Presets -->
    <div class="animate-fade-in" style="animation-delay: 50ms">
      <h2 class="section-title">Quick Start</h2>
      <div class="grid grid-cols-2 gap-3 mb-6">
        <button
          v-for="(preset, idx) in quickPresets" :key="preset.name"
          @click="startPreset(preset)"
          class="card-interactive p-4 text-left animate-fade-in"
          :style="{ animationDelay: `${100 + idx * 60}ms` }"
          :class="activePreset === preset.name ? 'ring-2 ring-bb-primary' : ''"
        >
          <p class="text-[10px] font-bold tracking-wider mb-1"
             :style="{ color: preset.accent || '#FF6B35' }">
            {{ preset.tag }}
          </p>
          <p class="text-sm font-semibold text-bb-text">{{ preset.name }}</p>
          <p class="text-xs text-bb-text-muted mt-1">{{ preset.desc }}</p>
        </button>
      </div>
    </div>

    <!-- Remote Control -->
    <div class="animate-fade-in" style="animation-delay: 200ms">
      <h2 class="section-title">Remote Control</h2>

      <!-- Home button — full width -->
      <button @click="sendCommand('navigate', { route: 'home' })"
              class="card-interactive w-full p-3 mb-3 flex items-center gap-3 active:scale-[0.98]">
        <div class="w-9 h-9 rounded-xl bg-bb-surface-light flex items-center justify-center flex-shrink-0">
          <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none"
               stroke="#58A6FF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/>
          </svg>
        </div>
        <div class="text-left">
          <p class="text-sm font-semibold text-bb-text">Return Home</p>
          <p class="text-[10px] text-bb-text-muted">Go back to welcome screen on GUI</p>
        </div>
      </button>

      <div class="grid grid-cols-2 gap-3 mb-6">
        <button @click="sendCommand('navigate', { route: 'training_select' })"
                class="card-interactive p-4 text-center active:scale-95">
          <div class="w-10 h-10 mx-auto mb-2 rounded-xl bg-bb-surface-light flex items-center justify-center">
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none"
                 stroke="#FF6B35" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z"/>
              <circle cx="12" cy="13" r="3"/>
            </svg>
          </div>
          <p class="text-xs font-semibold text-bb-text">Techniques</p>
          <p class="text-[10px] text-bb-text-muted">Open on GUI</p>
        </button>
        <button @click="sendCommand('navigate', { route: 'sparring_select' })"
                class="card-interactive p-4 text-center active:scale-95">
          <div class="w-10 h-10 mx-auto mb-2 rounded-xl bg-bb-surface-light flex items-center justify-center">
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none"
                 stroke="#FF5C5C" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <circle cx="12" cy="12" r="10"/><path d="M8 14s1.5 2 4 2 4-2 4-2"/><line x1="9" y1="9" x2="9.01" y2="9"/><line x1="15" y1="9" x2="15.01" y2="9"/>
            </svg>
          </div>
          <p class="text-xs font-semibold text-bb-text">Sparring</p>
          <p class="text-[10px] text-bb-text-muted">Open on GUI</p>
        </button>
        <button @click="sendCommand('navigate', { route: 'performance' })"
                class="card-interactive p-4 text-center active:scale-95">
          <div class="w-10 h-10 mx-auto mb-2 rounded-xl bg-bb-surface-light flex items-center justify-center">
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none"
                 stroke="#FFAB40" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>
            </svg>
          </div>
          <p class="text-xs font-semibold text-bb-text">Performance</p>
          <p class="text-[10px] text-bb-text-muted">Open on GUI</p>
        </button>
        <button @click="sendCommand('open_presets', {})"
                class="card-interactive p-4 text-center active:scale-95">
          <div class="w-10 h-10 mx-auto mb-2 rounded-xl bg-bb-surface-light flex items-center justify-center">
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none"
                 stroke="#BC8CFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <rect x="2" y="7" width="20" height="14" rx="2" ry="2"/><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/>
            </svg>
          </div>
          <p class="text-xs font-semibold text-bb-text">Quick Start</p>
          <p class="text-[10px] text-bb-text-muted">Open presets on GUI</p>
        </button>
      </div>
    </div>

    <!-- Status -->
    <transition name="fade">
      <div v-if="statusMsg" class="card text-center py-3 animate-fade-in">
        <p class="text-sm font-semibold" :class="statusOk ? 'text-bb-primary' : 'text-bb-danger'">
          {{ statusMsg }}
        </p>
      </div>
    </transition>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import * as api from '@/api/client'

const quickPresets = ref([])
const activePreset = ref('')
const statusMsg = ref('')
const statusOk = ref(true)

onMounted(async () => {
  try {
    quickPresets.value = await api.getRemotePresets()
  } catch {
    quickPresets.value = []
  }
})

async function startPreset(preset) {
  activePreset.value = preset.name
  await sendCommand('start_preset', preset)
  setTimeout(() => { activePreset.value = '' }, 2000)
}

async function sendCommand(action, config) {
  statusMsg.value = ''
  try {
    await api.sendRemoteCommand(action, config)
    statusOk.value = true
    statusMsg.value = action === 'start_preset'
      ? `Starting "${config.name}" on robot...`
      : 'Command sent to robot'
  } catch {
    statusOk.value = false
    statusMsg.value = 'Could not reach the robot'
  }
  setTimeout(() => { statusMsg.value = '' }, 3000)
}
</script>
