<template>
  <div class="pb-24 px-4 pt-6 max-w-lg mx-auto">
    <h1 class="text-2xl font-bold text-bb-text mb-1 animate-fade-in">Performance</h1>
    <p class="text-xs text-bb-text-muted mb-5 animate-fade-in">Track your progress over time</p>

    <!-- Date Range Selector -->
    <div class="flex gap-2 mb-4 animate-slide-up">
      <button
        v-for="range in timeRanges"
        :key="range.value"
        @click="changeRange(range.value)"
        class="px-3 py-1.5 rounded-lg text-xs font-semibold transition-all duration-200"
        :class="activeRange === range.value
          ? 'bg-bb-primary text-bb-bg shadow-sm shadow-bb-primary/20'
          : 'bg-bb-surface text-bb-text-secondary active:bg-bb-surface-light'"
      >
        {{ range.label }}
      </button>
    </div>

    <!-- Summary Cards -->
    <div v-if="trends && !loading" class="grid grid-cols-3 gap-2 mb-4 animate-slide-up" style="animation-delay: 30ms">
      <div class="card py-2 px-3 text-center">
        <p class="text-[10px] text-bb-text-muted uppercase tracking-wide">Sessions</p>
        <p class="text-lg font-bold text-bb-text mt-0.5">{{ trends.weekly_summary?.sessions || 0 }}</p>
        <p v-if="periodChangeWeek" class="text-[10px] font-semibold mt-0.5" :class="periodChangeColor(periodChangeWeek)">
          {{ periodChangeWeek }} vs last
        </p>
      </div>
      <div class="card py-2 px-3 text-center">
        <p class="text-[10px] text-bb-text-muted uppercase tracking-wide">Punches</p>
        <p class="text-lg font-bold text-bb-text mt-0.5">{{ (trends.weekly_summary?.total_punches || 0).toLocaleString() }}</p>
        <p v-if="periodChangeMonth" class="text-[10px] font-semibold mt-0.5" :class="periodChangeColor(periodChangeMonth)">
          {{ periodChangeMonth }} vs last
        </p>
      </div>
      <div class="card py-2 px-3 text-center">
        <p class="text-[10px] text-bb-text-muted uppercase tracking-wide">Avg Score</p>
        <p class="text-lg font-bold text-bb-text mt-0.5">{{ trends.weekly_summary?.avg_score || 0 }}</p>
        <p class="text-[10px] text-bb-text-muted mt-0.5">/100</p>
      </div>
    </div>

    <!-- Loading -->
    <div v-if="loading" class="space-y-4">
      <div class="skeleton h-12 w-full rounded-2xl" />
      <div class="skeleton h-56 w-full rounded-2xl" />
      <div class="skeleton h-56 w-full rounded-2xl" />
    </div>

    <div v-else>
      <!-- Chart Tab Selector -->
      <div class="flex gap-1 mb-4 bg-bb-surface rounded-xl p-1 animate-slide-up" style="animation-delay: 50ms">
        <button
          v-for="tab in chartTabs"
          :key="tab.id"
          @click="activeTab = tab.id"
          class="flex-1 px-2 py-2 rounded-lg text-[11px] font-semibold transition-all duration-200 text-center"
          :class="activeTab === tab.id
            ? 'bg-bb-surface-lighter text-bb-text shadow-sm'
            : 'text-bb-text-muted'"
        >
          {{ tab.label }}
        </button>
      </div>

      <!-- Punch Volume Chart -->
      <div v-show="activeTab === 'punches'" class="mb-4 animate-fade-in">
        <PunchChart
          title="Punches Per Session"
          type="line"
          :labels="chartLabels('punch_volume')"
          :datasets="[{
            data: chartValues('punch_volume'),
            label: 'Punches',
            fill: true,
            backgroundColor: 'rgba(0, 230, 118, 0.08)',
            borderColor: '#00E676',
            pointBackgroundColor: '#00E676',
            pointBorderColor: '#00E676',
          }]"
          :height="220"
        />
      </div>

      <!-- Reaction Time Chart -->
      <div v-show="activeTab === 'reaction'" class="mb-4 animate-fade-in">
        <PunchChart
          title="Reaction Time (lower is better)"
          type="line"
          :labels="chartLabels('reaction_time')"
          :datasets="[{
            data: chartValues('reaction_time'),
            label: 'Avg Reaction (ms)',
            borderColor: '#FF9800',
            backgroundColor: 'rgba(255, 152, 0, 0.08)',
            pointBackgroundColor: '#FF9800',
            pointBorderColor: '#FF9800',
            fill: true,
          }]"
          :height="220"
        />
      </div>

      <!-- Power Chart -->
      <div v-show="activeTab === 'power'" class="mb-4 animate-fade-in">
        <PunchChart
          title="Max Power Per Session"
          type="line"
          :labels="chartLabels('power')"
          :datasets="[{
            data: chartValues('power'),
            label: 'Power',
            borderColor: '#FF1744',
            backgroundColor: 'rgba(255, 23, 68, 0.08)',
            pointBackgroundColor: '#FF1744',
            pointBorderColor: '#FF1744',
            fill: true,
          }]"
          :height="220"
        />
        <div v-if="chartValues('power').length === 0" class="card text-center py-8 -mt-4">
          <p class="text-bb-text-muted text-sm">No power data yet</p>
          <p class="text-bb-text-muted text-xs mt-1">Complete power drills to see data here</p>
        </div>
      </div>

      <!-- Defense Chart -->
      <div v-show="activeTab === 'defense'" class="mb-4 animate-fade-in">
        <PunchChart
          title="Defense Rate"
          type="line"
          :labels="chartLabels('defense_rate')"
          :datasets="[{
            data: chartValues('defense_rate').map(v => Math.round(v * 100)),
            label: 'Defense Rate (%)',
            borderColor: '#42A5F5',
            backgroundColor: 'rgba(66, 165, 245, 0.08)',
            pointBackgroundColor: '#42A5F5',
            pointBorderColor: '#42A5F5',
            fill: true,
          }]"
          :height="220"
        />
        <div v-if="chartValues('defense_rate').length === 0" class="card text-center py-8 -mt-4">
          <p class="text-bb-text-muted text-sm">No defense data yet</p>
          <p class="text-bb-text-muted text-xs mt-1">Complete defence drills to track your blocks</p>
        </div>
      </div>

      <!-- Stamina Chart -->
      <div v-show="activeTab === 'stamina'" class="mb-4 animate-fade-in">
        <PunchChart
          title="Punches Per Minute"
          type="line"
          :labels="chartLabels('stamina')"
          :datasets="[{
            data: chartValues('stamina'),
            label: 'PPM',
            borderColor: '#AB47BC',
            backgroundColor: 'rgba(171, 71, 188, 0.08)',
            pointBackgroundColor: '#AB47BC',
            pointBorderColor: '#AB47BC',
            fill: true,
          }]"
          :height="220"
        />
        <div v-if="chartValues('stamina').length === 0" class="card text-center py-8 -mt-4">
          <p class="text-bb-text-muted text-sm">No stamina data yet</p>
          <p class="text-bb-text-muted text-xs mt-1">Complete stamina tests to see endurance trends</p>
        </div>
      </div>

      <!-- Personal Bests -->
      <div class="card mb-4 animate-slide-up" style="animation-delay: 150ms">
        <div class="flex items-center justify-between mb-3">
          <h3 class="section-title mb-0">Personal Bests</h3>
          <span class="text-[10px] text-bb-text-muted">All Time</span>
        </div>
        <div v-if="personalBests.length === 0" class="py-6 text-center">
          <div class="w-12 h-12 mx-auto mb-2 rounded-xl bg-bb-surface-light flex items-center justify-center">
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" class="text-bb-text-muted"><circle cx="12" cy="8" r="7"/><polyline points="8.21 13.89 7 23 12 20 17 23 15.79 13.88"/></svg>
          </div>
          <p class="text-bb-text-muted text-sm">No records set yet</p>
          <p class="text-bb-text-muted text-xs mt-1">Keep training to set personal records</p>
        </div>
        <div v-else class="space-y-2.5">
          <div
            v-for="(pr, idx) in personalBests"
            :key="idx"
            class="flex items-center gap-3 py-2 px-2 rounded-xl bg-bb-bg/40 border border-bb-border/10"
          >
            <div class="w-9 h-9 rounded-lg flex items-center justify-center text-xs font-bold" :class="pr.iconClass">
              {{ pr.icon }}
            </div>
            <div class="flex-1 min-w-0">
              <p class="text-sm font-medium text-bb-text">{{ pr.label }}</p>
              <p class="text-[10px] text-bb-text-muted">{{ pr.category }}</p>
            </div>
            <span class="text-sm font-bold tabular-nums" :class="pr.valueColor">{{ pr.value }}</span>
          </div>
        </div>
      </div>

      <!-- Population Comparison -->
      <div v-if="benchmarks && Object.keys(benchmarks.benchmarks || {}).length > 0" class="card mb-4 animate-slide-up" style="animation-delay: 200ms">
        <div class="flex items-center justify-between mb-3">
          <h3 class="section-title mb-0">vs Your Age/Gender Group</h3>
        </div>
        <div class="space-y-3">
          <div v-for="item in populationBars" :key="item.label">
            <div class="flex items-center justify-between mb-1">
              <span class="text-xs text-bb-text-secondary">{{ item.label }}</span>
              <span class="text-xs font-semibold" :class="item.color">{{ item.percentile }}th percentile</span>
            </div>
            <div class="relative h-2 bg-bb-surface-lighter rounded-full overflow-hidden">
              <div
                class="absolute left-0 top-0 h-full rounded-full transition-all duration-1000"
                :class="item.barColor"
                :style="{ width: `${item.percentile}%` }"
              />
              <!-- Population markers -->
              <div class="absolute top-0 h-full w-px bg-bb-text-muted/30" style="left: 25%" />
              <div class="absolute top-0 h-full w-px bg-bb-text-muted/30" style="left: 50%" />
              <div class="absolute top-0 h-full w-px bg-bb-text-muted/30" style="left: 75%" />
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import * as api from '@/api/client'
import PunchChart from '@/components/PunchChart.vue'

const activeRange = ref('30d')
const activeTab = ref('punches')
const loading = ref(true)
const trends = ref(null)
const benchmarks = ref(null)

const timeRanges = [
  { label: '7D', value: '7d' },
  { label: '30D', value: '30d' },
  { label: '90D', value: '90d' },
  { label: 'All', value: 'all' },
]

const chartTabs = [
  { id: 'punches', label: 'Volume' },
  { id: 'reaction', label: 'Reaction' },
  { id: 'power', label: 'Power' },
  { id: 'defense', label: 'Defense' },
  { id: 'stamina', label: 'Stamina' },
]

// Chart data from trends API
function chartLabels(key) {
  if (!trends.value || !trends.value[key]) return []
  return trends.value[key].map(p => {
    try {
      const d = new Date(p.date)
      return `${d.getMonth() + 1}/${d.getDate()}`
    } catch {
      return p.date
    }
  })
}

function chartValues(key) {
  if (!trends.value || !trends.value[key]) return []
  return trends.value[key].map(p => p.value)
}

// Period comparison
const periodChangeWeek = computed(() => trends.value?.period_comparison?.vs_last_week || null)
const periodChangeMonth = computed(() => trends.value?.period_comparison?.vs_last_month || null)

function periodChangeColor(val) {
  if (!val) return 'text-bb-text-muted'
  return val.startsWith('+') ? 'text-bb-primary' : val.startsWith('-') ? 'text-bb-danger' : 'text-bb-text-muted'
}

// Personal bests
const personalBests = computed(() => {
  if (!trends.value?.personal_bests) return []
  const pb = trends.value.personal_bests
  const bests = []

  if (pb.fastest_reaction_ms != null) {
    bests.push({
      label: 'Fastest Reaction',
      value: `${pb.fastest_reaction_ms}ms`,
      icon: 'R',
      iconClass: 'bg-bb-primary-dim text-bb-primary',
      valueColor: 'text-bb-primary',
      category: 'Speed',
    })
  }
  if (pb.most_punches != null) {
    bests.push({
      label: 'Most Punches',
      value: pb.most_punches.toLocaleString(),
      icon: 'P',
      iconClass: 'bg-bb-warning-dim text-bb-warning',
      valueColor: 'text-bb-warning',
      category: 'Volume',
    })
  }
  if (pb.best_defense_rate != null) {
    bests.push({
      label: 'Best Defense Rate',
      value: `${Math.round(pb.best_defense_rate * 100)}%`,
      icon: 'D',
      iconClass: 'bg-blue-500/20 text-blue-400',
      valueColor: 'text-blue-400',
      category: 'Defense',
    })
  }
  if (pb.max_power != null) {
    bests.push({
      label: 'Max Power',
      value: pb.max_power.toLocaleString(),
      icon: 'F',
      iconClass: 'bg-bb-danger-dim text-bb-danger',
      valueColor: 'text-bb-danger',
      category: 'Strength',
    })
  }
  if (pb.best_stamina_ppm != null) {
    bests.push({
      label: 'Best Punch Rate',
      value: `${pb.best_stamina_ppm} ppm`,
      icon: 'S',
      iconClass: 'bg-purple-500/20 text-purple-400',
      valueColor: 'text-purple-400',
      category: 'Endurance',
    })
  }

  return bests
})

// Population comparison bars
const populationBars = computed(() => {
  const b = benchmarks.value?.benchmarks
  if (!b) return []
  const bars = []
  for (const [key, val] of Object.entries(b)) {
    if (val && typeof val === 'object' && val.percentile != null) {
      const p = Math.min(99, Math.max(1, val.percentile))
      let color = 'text-bb-text-secondary'
      let barColor = 'bg-bb-text-muted'
      if (p >= 90) { color = 'text-bb-primary'; barColor = 'bg-bb-primary' }
      else if (p >= 75) { color = 'text-blue-400'; barColor = 'bg-blue-400' }
      else if (p >= 50) { color = 'text-bb-text-secondary'; barColor = 'bg-bb-text-secondary' }
      else if (p >= 25) { color = 'text-bb-warning'; barColor = 'bg-bb-warning' }
      else { color = 'text-bb-danger'; barColor = 'bg-bb-danger' }

      bars.push({
        label: key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
        percentile: p,
        color,
        barColor,
      })
    }
  }
  return bars
})

async function loadData() {
  loading.value = true
  try {
    trends.value = await api.getSessionTrends(activeRange.value)
  } catch (e) {
    trends.value = null
  } finally {
    loading.value = false
  }
}

function changeRange(range) {
  activeRange.value = range
  loadData()
}

onMounted(async () => {
  await loadData()
  // Load benchmarks in parallel
  api.getBenchmarks().then(b => { benchmarks.value = b }).catch(() => {})
})
</script>
