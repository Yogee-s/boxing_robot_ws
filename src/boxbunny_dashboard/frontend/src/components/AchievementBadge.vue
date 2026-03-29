<template>
  <div
    class="flex flex-col items-center gap-2 p-3 rounded-xl transition-all duration-300"
    :class="unlocked ? 'bg-bb-surface-light' : 'bg-bb-surface opacity-40'"
  >
    <div
      class="w-12 h-12 rounded-xl flex items-center justify-center transition-transform duration-300 overflow-hidden"
      :class="[
        unlocked ? iconBg : 'bg-bb-surface-lighter',
        unlocked ? 'scale-100' : 'scale-90',
      ]"
    >
      <img
        v-if="unlocked"
        :src="`/achievements/${achievementId}.svg`"
        :alt="name"
        class="w-8 h-8"
      />
      <span v-else class="text-xl text-bb-text-muted">?</span>
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
  first_blood: { name: 'First Blood', bg: 'bg-bb-primary-dim' },
  century: { name: 'Century', bg: 'bg-blue-500/20' },
  fury: { name: 'Fury', bg: 'bg-bb-danger-dim' },
  thousand_fists: { name: '1000 Fists', bg: 'bg-purple-500/20' },
  speed_demon: { name: 'Speed Demon', bg: 'bg-yellow-500/20' },
  weekly_warrior: { name: 'Weekly Warrior', bg: 'bg-bb-warning-dim' },
  consistent: { name: 'Consistent', bg: 'bg-bb-primary-dim' },
  iron_chin: { name: 'Iron Chin', bg: 'bg-gray-500/20' },
  marathon: { name: 'Marathon', bg: 'bg-blue-500/20' },
  centurion: { name: 'Centurion', bg: 'bg-bb-warning-dim' },
  well_rounded: { name: 'Well Rounded', bg: 'bg-purple-500/20' },
  perfect_round: { name: 'Perfect Round', bg: 'bg-bb-primary-dim' },
}

const meta = computed(() => achievementMeta[props.achievementId] || {
  name: props.achievementId,
  bg: 'bg-bb-surface-lighter',
})

const name = computed(() => meta.value.name)
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
