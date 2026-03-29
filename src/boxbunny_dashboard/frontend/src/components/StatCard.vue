<template>
  <div class="card animate-fade-in" :style="{ animationDelay: `${delay}ms` }">
    <div class="flex items-start justify-between mb-1">
      <span class="text-bb-text-muted text-xs font-medium uppercase tracking-wide">
        {{ label }}
      </span>
      <div
        v-if="icon"
        class="w-8 h-8 rounded-lg flex items-center justify-center text-sm"
        :class="iconBgClass"
      >
        {{ icon }}
      </div>
    </div>
    <div class="flex items-baseline gap-1.5">
      <span class="text-2xl font-bold text-bb-text tabular-nums">
        {{ displayValue }}
      </span>
      <span v-if="unit" class="text-sm text-bb-text-secondary">{{ unit }}</span>
    </div>
    <div v-if="subtitle" class="text-xs text-bb-text-muted mt-1">
      {{ subtitle }}
    </div>
    <div v-if="change !== null" class="flex items-center gap-1 mt-1.5">
      <span
        class="text-xs font-semibold"
        :class="change >= 0 ? 'text-bb-primary' : 'text-bb-danger'"
      >
        {{ change >= 0 ? '+' : '' }}{{ change }}%
      </span>
      <span class="text-[10px] text-bb-text-muted">vs last week</span>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  label: { type: String, required: true },
  value: { type: [Number, String], default: 0 },
  unit: { type: String, default: '' },
  icon: { type: String, default: '' },
  subtitle: { type: String, default: '' },
  change: { type: Number, default: null },
  color: { type: String, default: 'green' },
  delay: { type: Number, default: 0 },
})

const displayValue = computed(() => {
  if (typeof props.value === 'number') {
    return props.value.toLocaleString()
  }
  return props.value
})

const iconBgClass = computed(() => {
  const map = {
    green: 'bg-bb-primary-dim text-bb-primary',
    warning: 'bg-bb-warning-dim text-bb-warning',
    danger: 'bg-bb-danger-dim text-bb-danger',
    neutral: 'bg-bb-surface-lighter text-bb-text-secondary',
  }
  return map[props.color] || map.green
})
</script>
