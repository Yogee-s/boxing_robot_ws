<template>
  <div class="pb-24 px-4 pt-6 max-w-lg mx-auto">
    <div class="flex items-center justify-between mb-5 animate-fade-in">
      <h1 class="text-2xl font-bold text-bb-text">Presets</h1>
      <button @click="showCreateModal = true" class="btn-primary text-sm py-2 px-4">
        + New
      </button>
    </div>

    <!-- Loading -->
    <div v-if="loading" class="space-y-3">
      <div v-for="i in 4" :key="i" class="skeleton h-20 w-full rounded-2xl" />
    </div>

    <!-- Empty -->
    <div v-else-if="presets.length === 0" class="card text-center py-16">
      <div class="text-4xl mb-3 text-bb-text-muted">--</div>
      <p class="text-bb-text-muted text-sm">No presets yet</p>
      <p class="text-bb-text-muted text-xs mt-1">Create a preset to save your favorite configurations</p>
      <button @click="showCreateModal = true" class="btn-primary mt-4 text-sm">
        Create Preset
      </button>
    </div>

    <!-- Preset List -->
    <div v-else class="space-y-3">
      <div
        v-for="(preset, idx) in sortedPresets"
        :key="preset.id"
        class="card-interactive animate-fade-in"
        :style="{ animationDelay: `${idx * 50}ms` }"
      >
        <div class="flex items-start justify-between">
          <div class="flex-1 min-w-0">
            <div class="flex items-center gap-2 mb-1">
              <span :class="typeBadgeClass(preset.preset_type)" class="badge text-[10px]">
                {{ preset.preset_type }}
              </span>
              <span v-if="preset.use_count > 0" class="text-[10px] text-bb-text-muted">
                Used {{ preset.use_count }}x
              </span>
            </div>
            <p class="text-sm font-semibold text-bb-text truncate">{{ preset.name }}</p>
            <p v-if="preset.description" class="text-xs text-bb-text-secondary mt-0.5 line-clamp-2">
              {{ preset.description }}
            </p>
          </div>
          <div class="flex items-center gap-2 ml-3">
            <button
              @click.stop="toggleFavorite(preset.id)"
              class="w-8 h-8 rounded-lg flex items-center justify-center transition-colors"
              :class="preset.is_favorite ? 'text-bb-warning' : 'text-bb-text-muted'"
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24"
                   :fill="preset.is_favorite ? 'currentColor' : 'none'"
                   stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
              </svg>
            </button>
            <button
              @click.stop="deletePresetById(preset.id)"
              class="w-8 h-8 rounded-lg flex items-center justify-center text-bb-text-muted hover:text-bb-danger transition-colors"
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none"
                   stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polyline points="3 6 5 6 21 6" />
                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
              </svg>
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- Create Modal -->
    <Teleport to="body">
      <transition name="fade">
        <div
          v-if="showCreateModal"
          class="fixed inset-0 z-50 bg-black/60 flex items-end justify-center"
          @click.self="showCreateModal = false"
        >
          <transition name="slide-up">
            <div
              v-if="showCreateModal"
              class="bg-bb-surface rounded-t-3xl w-full max-w-lg p-6 safe-bottom"
            >
              <div class="w-10 h-1 bg-bb-border/50 rounded-full mx-auto mb-5" />
              <h2 class="text-lg font-bold text-bb-text mb-4">New Preset</h2>

              <form @submit.prevent="createNewPreset" class="space-y-4">
                <div>
                  <label class="block text-xs font-medium text-bb-text-secondary mb-1.5">Name</label>
                  <input v-model="newPreset.name" type="text" class="input" placeholder="Preset name" required />
                </div>
                <div>
                  <label class="block text-xs font-medium text-bb-text-secondary mb-1.5">Type</label>
                  <div class="grid grid-cols-3 gap-2">
                    <button
                      v-for="t in presetTypes"
                      :key="t"
                      type="button"
                      @click="newPreset.preset_type = t"
                      class="py-2 rounded-xl text-xs font-semibold border transition-all"
                      :class="newPreset.preset_type === t
                        ? 'border-bb-green bg-bb-green-dim text-bb-green'
                        : 'border-bb-border/50 bg-bb-surface-light text-bb-text-secondary'"
                    >
                      {{ t }}
                    </button>
                  </div>
                </div>
                <div>
                  <label class="block text-xs font-medium text-bb-text-secondary mb-1.5">Description</label>
                  <input v-model="newPreset.description" type="text" class="input" placeholder="Optional description" />
                </div>
                <div class="flex gap-3 pt-2">
                  <button type="button" @click="showCreateModal = false" class="btn-secondary flex-1">
                    Cancel
                  </button>
                  <button type="submit" :disabled="!newPreset.name" class="btn-primary flex-1">
                    Create
                  </button>
                </div>
              </form>
            </div>
          </transition>
        </div>
      </transition>
    </Teleport>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import * as api from '@/api/client'

const presets = ref([])
const loading = ref(true)
const showCreateModal = ref(false)

const presetTypes = ['reaction', 'shadow', 'defence']

const newPreset = ref({
  name: '',
  preset_type: 'reaction',
  description: '',
  config_json: '{}',
  tags: '',
})

const sortedPresets = computed(() => {
  return [...presets.value]
    .filter(p => p.tags !== 'archived')
    .sort((a, b) => {
      if (a.is_favorite && !b.is_favorite) return -1
      if (!a.is_favorite && b.is_favorite) return 1
      return 0
    })
})

function typeBadgeClass(type) {
  const map = {
    reaction: 'badge-green',
    shadow: 'bg-purple-500/20 text-purple-400',
    defence: 'badge-warning',
  }
  return map[type] || 'badge-neutral'
}

async function fetchPresets() {
  loading.value = true
  try {
    presets.value = await api.getPresets()
  } catch (e) {
    console.error('Failed to fetch presets:', e)
  } finally {
    loading.value = false
  }
}

async function createNewPreset() {
  try {
    const created = await api.createPreset({
      name: newPreset.value.name,
      preset_type: newPreset.value.preset_type,
      description: newPreset.value.description,
      config_json: newPreset.value.config_json,
      tags: newPreset.value.tags,
    })
    presets.value.push(created)
    showCreateModal.value = false
    newPreset.value = { name: '', preset_type: 'reaction', description: '', config_json: '{}', tags: '' }
  } catch (e) {
    console.error('Failed to create preset:', e)
  }
}

async function toggleFavorite(presetId) {
  try {
    const result = await api.togglePresetFavorite(presetId)
    const preset = presets.value.find(p => p.id === presetId)
    if (preset) preset.is_favorite = result.is_favorite
  } catch (e) {
    console.error('Failed to toggle favorite:', e)
  }
}

async function deletePresetById(presetId) {
  try {
    await api.deletePreset(presetId)
    presets.value = presets.value.filter(p => p.id !== presetId)
  } catch (e) {
    console.error('Failed to delete preset:', e)
  }
}

onMounted(fetchPresets)
</script>
