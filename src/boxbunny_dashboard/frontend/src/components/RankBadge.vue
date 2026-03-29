<template>
  <div class="inline-flex items-center gap-2" :class="sizeClass">
    <div
      class="flex items-center justify-center rounded-lg font-bold"
      :class="[rankColorClass, iconSizeClass]"
    >
      {{ rankIcon }}
    </div>
    <div v-if="showLabel" class="flex flex-col">
      <span class="font-semibold text-bb-text" :class="labelSizeClass">{{ rank }}</span>
      <span v-if="showXp" class="text-bb-text-muted" :class="xpSizeClass">
        {{ xp.toLocaleString() }} XP
      </span>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  rank: { type: String, default: 'Novice' },
  xp: { type: Number, default: 0 },
  size: { type: String, default: 'md' },
  showLabel: { type: Boolean, default: true },
  showXp: { type: Boolean, default: false },
})

const rankIcons = {
  'Novice': 'N',
  'Contender': 'C',
  'Fighter': 'F',
  'Warrior': 'W',
  'Champion': 'CH',
  'Elite': 'E',
}

const rankColors = {
  'Novice': 'bg-gray-600/30 text-gray-400',
  'Contender': 'bg-blue-600/30 text-blue-400',
  'Fighter': 'bg-bb-primary-dim text-bb-primary',
  'Warrior': 'bg-purple-600/30 text-purple-400',
  'Champion': 'bg-bb-warning-dim text-bb-warning',
  'Elite': 'bg-bb-danger-dim text-bb-danger',
}

const rankIcon = computed(() => rankIcons[props.rank] || 'N')
const rankColorClass = computed(() => rankColors[props.rank] || rankColors['Novice'])

const sizeClass = computed(() => {
  const map = { sm: '', md: '', lg: 'gap-3' }
  return map[props.size] || ''
})

const iconSizeClass = computed(() => {
  const map = { sm: 'w-6 h-6 text-[10px]', md: 'w-8 h-8 text-xs', lg: 'w-12 h-12 text-base' }
  return map[props.size] || map.md
})

const labelSizeClass = computed(() => {
  const map = { sm: 'text-xs', md: 'text-sm', lg: 'text-lg' }
  return map[props.size] || map.md
})

const xpSizeClass = computed(() => {
  const map = { sm: 'text-[10px]', md: 'text-xs', lg: 'text-sm' }
  return map[props.size] || map.md
})
</script>
