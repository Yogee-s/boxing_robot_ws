<template>
  <div
    class="flex flex-col items-center gap-2 p-3 rounded-xl transition-all duration-300"
    :class="unlocked ? 'bg-bb-surface-light' : 'bg-bb-surface opacity-40'"
  >
    <div
      class="w-12 h-12 rounded-xl flex items-center justify-center text-xl transition-transform duration-300"
      :class="[
        unlocked ? iconBg : 'bg-bb-surface-lighter text-bb-text-muted',
        unlocked ? 'scale-100' : 'scale-90',
      ]"
    >
      {{ unlocked ? iconText : '?' }}
    </div>
    <div class="text-center">
      <p
        class="text-xs font-semibold leading-tight"
        :class="unlocked ? 'text-bb-text' : 'text-bb-text-muted'"
      >
        {{ unlocked ? name : '???' }}
      </p>
      <p v-if="unlocked && unlockedAt" class="text-[10px] text-bb-text-muted mt-0.5">
        {{ formattedDate }}
      </p>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  achievementId: { type: String, required: true },
  unlocked: { type: Boolean, default: false },
  unlockedAt: { type: String, default: '' },
})

const achievementMeta = {
  first_blood: { name: 'First Blood', icon: 'I', bg: 'bg-bb-primary-dim text-bb-primary' },
  century: { name: 'Century', icon: 'C', bg: 'bg-blue-500/20 text-blue-400' },
  fury: { name: 'Fury', icon: 'F', bg: 'bg-bb-danger-dim text-bb-danger' },
  thousand_fists: { name: '1000 Fists', icon: 'K', bg: 'bg-purple-500/20 text-purple-400' },
  speed_demon: { name: 'Speed Demon', icon: 'S', bg: 'bg-yellow-500/20 text-yellow-400' },
  weekly_warrior: { name: 'Weekly Warrior', icon: 'W', bg: 'bg-bb-warning-dim text-bb-warning' },
  consistent: { name: 'Consistent', icon: '30', bg: 'bg-bb-primary-dim text-bb-primary' },
  iron_chin: { name: 'Iron Chin', icon: '10', bg: 'bg-gray-500/20 text-gray-400' },
  marathon: { name: 'Marathon', icon: '50', bg: 'bg-blue-500/20 text-blue-400' },
  centurion: { name: 'Centurion', icon: 'C', bg: 'bg-bb-warning-dim text-bb-warning' },
  well_rounded: { name: 'Well Rounded', icon: 'R', bg: 'bg-purple-500/20 text-purple-400' },
  perfect_round: { name: 'Perfect Round', icon: 'P', bg: 'bg-bb-primary-dim text-bb-primary' },
}

const meta = computed(() => achievementMeta[props.achievementId] || {
  name: props.achievementId,
  icon: '?',
  bg: 'bg-bb-surface-lighter text-bb-text-secondary',
})

const name = computed(() => meta.value.name)
const iconText = computed(() => meta.value.icon)
const iconBg = computed(() => meta.value.bg)

const formattedDate = computed(() => {
  if (!props.unlockedAt) return ''
  try {
    return new Date(props.unlockedAt).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
    })
  } catch {
    return props.unlockedAt
  }
})
</script>
