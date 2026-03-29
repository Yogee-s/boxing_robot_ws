<template>
  <div class="flex flex-col items-center gap-3">
    <div
      ref="gridEl"
      class="relative touch-none select-none"
      :style="{ width: `${gridSize}px`, height: `${gridSize}px` }"
      @pointerdown="onPointerDown"
      @pointermove="onPointerMove"
      @pointerup="onPointerUp"
      @pointercancel="onPointerUp"
    >
      <!-- SVG lines connecting selected dots -->
      <svg class="absolute inset-0 pointer-events-none" :width="gridSize" :height="gridSize">
        <line
          v-for="(seg, idx) in lineSegments"
          :key="idx"
          :x1="seg.x1" :y1="seg.y1" :x2="seg.x2" :y2="seg.y2"
          :stroke="error ? '#E53935' : '#FF6B35'"
          stroke-width="3"
          stroke-linecap="round"
          :opacity="0.7"
        />
        <!-- Active line from last dot to current finger -->
        <line
          v-if="drawing && selected.length > 0 && currentPos"
          :x1="dotCenter(selected[selected.length - 1]).x"
          :y1="dotCenter(selected[selected.length - 1]).y"
          :x2="currentPos.x"
          :y2="currentPos.y"
          stroke="#FF6B35"
          stroke-width="2"
          stroke-linecap="round"
          opacity="0.4"
        />
      </svg>

      <!-- Dots -->
      <div
        v-for="dot in 9"
        :key="dot"
        class="absolute flex items-center justify-center transition-transform duration-150"
        :style="dotStyle(dot)"
      >
        <div
          class="rounded-full transition-all duration-200"
          :class="dotClass(dot)"
          :style="dotInnerStyle(dot)"
        />
      </div>
    </div>

    <!-- Status -->
    <div class="flex items-center justify-between w-full px-1">
      <span class="text-[11px]" :class="error ? 'text-bb-danger' : 'text-bb-text-muted'">
        {{ statusText }}
      </span>
      <button
        v-if="selected.length > 0 && !drawing"
        @click="reset"
        class="text-xs text-bb-text-secondary active:opacity-70"
      >
        Reset
      </button>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'

const props = defineProps({
  minDots: { type: Number, default: 4 },
  size: { type: Number, default: 240 },
  error: { type: Boolean, default: false },
})

const emit = defineEmits(['update:pattern', 'complete'])

const gridEl = ref(null)
const selected = ref([])
const drawing = ref(false)
const currentPos = ref(null)
const gridSize = computed(() => props.size)
const dotSpacing = computed(() => props.size / 3)
const dotRadius = 22
const hitRadius = 34

function dotCenter(dot) {
  const row = Math.floor((dot - 1) / 3)
  const col = (dot - 1) % 3
  return {
    x: col * dotSpacing.value + dotSpacing.value / 2,
    y: row * dotSpacing.value + dotSpacing.value / 2,
  }
}

function dotStyle(dot) {
  const c = dotCenter(dot)
  return {
    left: `${c.x - dotRadius}px`,
    top: `${c.y - dotRadius}px`,
    width: `${dotRadius * 2}px`,
    height: `${dotRadius * 2}px`,
  }
}

function dotInnerStyle(dot) {
  const isSelected = selected.value.includes(dot)
  const size = isSelected ? 20 : 10
  return { width: `${size}px`, height: `${size}px` }
}

function dotClass(dot) {
  if (props.error && selected.value.includes(dot)) {
    return 'bg-bb-danger shadow-sm shadow-bb-danger/40'
  }
  if (selected.value.includes(dot)) {
    return 'bg-bb-primary shadow-sm shadow-bb-primary/40'
  }
  return 'bg-bb-text-muted/60'
}

const lineSegments = computed(() => {
  const segs = []
  for (let i = 1; i < selected.value.length; i++) {
    const from = dotCenter(selected.value[i - 1])
    const to = dotCenter(selected.value[i])
    segs.push({ x1: from.x, y1: from.y, x2: to.x, y2: to.y })
  }
  return segs
})

const statusText = computed(() => {
  if (props.error) return 'Try again'
  if (selected.value.length === 0) return `Connect at least ${props.minDots} dots`
  if (selected.value.length < props.minDots) {
    return `${selected.value.length} / ${props.minDots}+ dots`
  }
  return `${selected.value.length} dots selected`
})

function getRelativePos(e) {
  if (!gridEl.value) return null
  const rect = gridEl.value.getBoundingClientRect()
  return {
    x: e.clientX - rect.left,
    y: e.clientY - rect.top,
  }
}

function hitTestDot(pos) {
  for (let dot = 1; dot <= 9; dot++) {
    const c = dotCenter(dot)
    const dx = pos.x - c.x
    const dy = pos.y - c.y
    if (Math.sqrt(dx * dx + dy * dy) <= hitRadius) {
      return dot
    }
  }
  return null
}

function onPointerDown(e) {
  if (selected.value.length > 0 && !drawing.value) {
    // Reset if tapping again after completion
    selected.value = []
  }
  drawing.value = true
  currentPos.value = getRelativePos(e)
  gridEl.value?.setPointerCapture(e.pointerId)
  const dot = hitTestDot(currentPos.value)
  if (dot) {
    selected.value = [dot]
    emit('update:pattern', [...selected.value])
  }
}

function onPointerMove(e) {
  if (!drawing.value) return
  currentPos.value = getRelativePos(e)
  const dot = hitTestDot(currentPos.value)
  if (dot && !selected.value.includes(dot)) {
    selected.value.push(dot)
    emit('update:pattern', [...selected.value])
  }
}

function onPointerUp() {
  drawing.value = false
  currentPos.value = null
  if (selected.value.length >= props.minDots) {
    emit('complete', [...selected.value])
  }
  emit('update:pattern', [...selected.value])
}

function reset() {
  selected.value = []
  drawing.value = false
  currentPos.value = null
  emit('update:pattern', [])
}

defineExpose({ reset })
</script>
