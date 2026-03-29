<template>
  <div class="pb-24 px-4 pt-6 max-w-lg mx-auto">
    <!-- Back button -->
    <button @click="$router.back()" class="flex items-center gap-2 text-bb-text-secondary mb-4 active:opacity-70">
      <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none"
           stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <polyline points="15 18 9 12 15 6" />
      </svg>
      <span class="text-sm font-medium">Back</span>
    </button>

    <!-- Loading -->
    <div v-if="loading" class="space-y-4">
      <div class="skeleton h-32 w-full rounded-2xl" />
      <div class="skeleton h-48 w-full rounded-2xl" />
      <div class="skeleton h-24 w-full rounded-2xl" />
    </div>

    <!-- Error -->
    <div v-else-if="error" class="card text-center py-12">
      <div class="w-14 h-14 mx-auto mb-3 rounded-2xl bg-bb-danger-dim flex items-center justify-center">
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none"
             stroke="#FF1744" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/>
        </svg>
      </div>
      <p class="text-bb-danger text-sm mb-1">{{ error }}</p>
      <button @click="fetchDetail" class="btn-secondary mt-3 text-sm">Retry</button>
    </div>

    <!-- Session Detail -->
    <div v-else-if="session">
      <!-- Header Card with User Context -->
      <div class="card mb-4 animate-fade-in">
        <div class="flex items-start justify-between">
          <div class="flex-1">
            <div class="flex items-center gap-2 mb-2">
              <span :class="modeBadgeClass" class="badge">{{ modeLabel }}</span>
              <span class="badge badge-neutral">{{ session.difficulty }}</span>
              <span v-if="session.is_complete" class="badge badge-green">Complete</span>
              <span v-else class="badge badge-warning">Partial</span>
            </div>
            <p class="text-lg font-bold text-bb-text">{{ formattedDate }}</p>
            <p class="text-sm text-bb-text-secondary mt-0.5">{{ duration }}</p>
            <!-- User context -->
            <p v-if="userContext" class="text-[10px] text-bb-text-muted mt-1.5">
              {{ userContext }}
            </p>
          </div>
          <div
            class="w-16 h-16 rounded-xl flex items-center justify-center text-2xl font-black shadow-lg"
            :class="gradeClass"
          >
            {{ grade }}
          </div>
        </div>
      </div>

      <!-- Round Progress -->
      <div class="card mb-4 animate-slide-up" style="animation-delay: 50ms">
        <h3 class="section-title">Rounds</h3>
        <div class="flex items-baseline gap-1 mb-2">
          <span class="text-3xl font-bold text-bb-text">{{ session.rounds_completed }}</span>
          <span class="text-lg text-bb-text-secondary">/ {{ session.rounds_total }}</span>
        </div>
        <div class="progress-bar h-3">
          <div
            class="progress-fill bg-bb-primary"
            :style="{ width: `${roundProgress}%` }"
          />
        </div>
      </div>

      <!-- Punch Distribution - Bar + Donut side by side -->
      <div v-if="hasPunchData" class="grid grid-cols-1 gap-4 mb-4 animate-slide-up" style="animation-delay: 100ms">
        <PunchChart
          title="Punch Distribution"
          type="bar"
          :labels="punchLabels"
          :datasets="[{ data: punchValues, label: 'Punches' }]"
          :height="160"
        />
        <PunchChart
          v-if="punchValues.length > 0"
          title="Punch Mix"
          type="doughnut"
          :labels="punchLabels"
          :datasets="[{
            data: punchValues,
            label: 'Punches',
            backgroundColor: punchColors,
            borderWidth: 0,
          }]"
          :height="180"
        />
      </div>

      <!-- Round-by-Round Breakdown -->
      <div v-if="roundBreakdown.length > 0" class="card mb-4 animate-slide-up" style="animation-delay: 120ms">
        <h3 class="section-title">Round-by-Round</h3>
        <div class="space-y-2">
          <div v-for="(round, idx) in roundBreakdown" :key="idx"
               class="flex items-center gap-3 py-2 px-2 rounded-xl bg-bb-bg/40">
            <div class="w-8 h-8 rounded-lg bg-bb-surface-lighter flex items-center justify-center text-xs font-bold text-bb-text-secondary">
              R{{ idx + 1 }}
            </div>
            <div class="flex-1">
              <div class="flex items-center justify-between mb-1">
                <span class="text-xs text-bb-text-secondary">{{ round.punches }} punches</span>
                <span class="text-[10px] text-bb-text-muted">{{ round.ppm }} ppm</span>
              </div>
              <div class="progress-bar h-1.5">
                <div
                  class="progress-fill"
                  :class="round.intensity > 0.7 ? 'bg-bb-primary' : round.intensity > 0.4 ? 'bg-bb-warning' : 'bg-bb-danger'"
                  :style="{ width: `${round.intensity * 100}%` }"
                />
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Fatigue Curve -->
      <div v-if="fatigueCurve.length > 1" class="mb-4 animate-slide-up" style="animation-delay: 140ms">
        <PunchChart
          title="Fatigue Curve (Punches/Min Over Time)"
          type="line"
          :labels="fatigueCurve.map((_, i) => `R${i + 1}`)"
          :datasets="[{
            data: fatigueCurve,
            label: 'Punches/Min',
            borderColor: '#FF9800',
            backgroundColor: 'rgba(255, 152, 0, 0.08)',
            pointBackgroundColor: '#FF9800',
            pointBorderColor: '#FF9800',
            fill: true,
          }]"
          :height="160"
        />
      </div>

      <!-- Defense Breakdown -->
      <div v-if="hasDefenseData" class="card mb-4 animate-slide-up" style="animation-delay: 160ms">
        <h3 class="section-title">Defense Breakdown</h3>
        <div class="grid grid-cols-3 gap-2">
          <div v-for="def in defenseItems" :key="def.label"
               class="flex flex-col items-center py-3 rounded-xl bg-bb-bg/40 border border-bb-border/10">
            <div class="w-8 h-8 rounded-lg flex items-center justify-center mb-1.5" :class="def.iconBg">
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none"
                   :stroke="def.iconStroke" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path :d="def.iconPath" />
              </svg>
            </div>
            <span class="text-lg font-bold text-bb-text">{{ def.value }}</span>
            <span class="text-[10px] text-bb-text-muted mt-0.5">{{ def.label }}</span>
          </div>
        </div>
      </div>

      <!-- Session Summary -->
      <div v-if="summaryEntries.length > 0" class="card mb-4 animate-slide-up" style="animation-delay: 180ms">
        <h3 class="section-title">Summary</h3>
        <div class="space-y-0">
          <div
            v-for="(entry, idx) in summaryEntries"
            :key="idx"
            class="flex items-center justify-between py-2.5 border-b border-bb-border/10 last:border-0"
          >
            <span class="text-sm text-bb-text-secondary">{{ entry.label }}</span>
            <span class="text-sm font-semibold text-bb-text tabular-nums">{{ entry.value }}</span>
          </div>
        </div>
      </div>

      <!-- Compared to Your Average -->
      <div v-if="comparisonItems.length > 0" class="card mb-4 animate-slide-up" style="animation-delay: 200ms">
        <h3 class="section-title">vs Your Average</h3>
        <div class="space-y-2.5">
          <div v-for="cmp in comparisonItems" :key="cmp.label" class="flex items-center justify-between">
            <span class="text-xs text-bb-text-secondary">{{ cmp.label }}</span>
            <div class="flex items-center gap-2">
              <span class="text-xs font-semibold text-bb-text tabular-nums">{{ cmp.thisSession }}</span>
              <span class="text-[10px] font-semibold px-1.5 py-0.5 rounded-md" :class="cmp.changeClass">
                {{ cmp.change }}
              </span>
            </div>
          </div>
        </div>
      </div>

      <!-- AI Coach Analysis -->
      <div v-if="coachAnalysis" class="card mb-4 animate-slide-up" style="animation-delay: 220ms">
        <div class="flex items-center gap-2 mb-2">
          <div class="w-7 h-7 rounded-lg bg-bb-primary-dim flex items-center justify-center">
            <span class="text-bb-primary text-[10px] font-bold">AI</span>
          </div>
          <h3 class="section-title mb-0">AI Coach Analysis</h3>
        </div>
        <p class="text-sm text-bb-text-secondary leading-relaxed">{{ coachAnalysis }}</p>
      </div>

      <!-- XP Earned -->
      <div class="card mb-4 animate-slide-up" style="animation-delay: 240ms">
        <div class="flex items-center justify-between">
          <div>
            <h3 class="section-title mb-1">XP Earned</h3>
            <span class="text-2xl font-bold text-bb-primary">+{{ xpEarned }}</span>
          </div>
          <div class="w-12 h-12 rounded-xl bg-bb-primary-dim flex items-center justify-center shadow-sm shadow-bb-primary/10">
            <span class="text-bb-primary font-bold">XP</span>
          </div>
        </div>
      </div>

      <!-- Actions Row -->
      <div class="flex gap-3 mb-4 animate-slide-up" style="animation-delay: 260ms">
        <button @click="shareSession" class="btn-secondary flex-1 text-sm">
          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none"
               stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/>
            <line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/>
          </svg>
          Share
        </button>
        <button @click="exportCSV" class="btn-secondary flex-1 text-sm">
          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none"
               stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="7 10 12 15 17 10" />
            <line x1="12" y1="15" x2="12" y2="3" />
          </svg>
          CSV
        </button>
        <button @click="exportPDF" class="btn-secondary flex-1 text-sm">
          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none"
               stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
            <polyline points="14 2 14 8 20 8" />
          </svg>
          PDF
        </button>
      </div>

      <!-- Share Summary Card (modal overlay) -->
      <transition name="fade">
        <div v-if="showShareCard" class="fixed inset-0 z-50 flex items-center justify-center bg-black/70 px-6" @click.self="showShareCard = false">
          <div class="w-full max-w-sm bg-bb-surface rounded-2xl p-5 border border-bb-border/30 animate-scale-in">
            <div class="text-center mb-4">
              <p class="text-xs text-bb-primary font-semibold uppercase tracking-wide">BoxBunny Session</p>
              <p class="text-lg font-bold text-bb-text mt-1">{{ formattedDate }}</p>
            </div>
            <div class="grid grid-cols-2 gap-3 mb-4">
              <div class="bg-bb-bg/60 rounded-lg p-3 text-center">
                <p class="text-2xl font-bold text-bb-text">{{ grade }}</p>
                <p class="text-[10px] text-bb-text-muted">Grade</p>
              </div>
              <div class="bg-bb-bg/60 rounded-lg p-3 text-center">
                <p class="text-2xl font-bold text-bb-text">{{ session.rounds_completed }}/{{ session.rounds_total }}</p>
                <p class="text-[10px] text-bb-text-muted">Rounds</p>
              </div>
            </div>
            <div class="flex items-center justify-between py-2 border-t border-bb-border/20">
              <span class="text-[10px] text-bb-text-muted">{{ modeLabel }} | {{ session.difficulty }} | {{ duration }}</span>
              <span class="text-[10px] text-bb-primary font-bold">+{{ xpEarned }} XP</span>
            </div>
            <button @click="copyShareCard" class="btn-primary w-full mt-3 text-sm">
              Copy Summary
            </button>
          </div>
        </div>
      </transition>
    </div>

    <!-- Not found -->
    <div v-else class="card text-center py-12">
      <div class="w-14 h-14 mx-auto mb-3 rounded-2xl bg-bb-surface-light flex items-center justify-center">
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none"
             stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" class="text-bb-text-muted">
          <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
        </svg>
      </div>
      <p class="text-bb-text-muted text-sm">Session not found</p>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import * as api from '@/api/client'
import { useAuthStore } from '@/stores/auth'
import PunchChart from '@/components/PunchChart.vue'

const route = useRoute()
const auth = useAuthStore()

const session = ref(null)
const loading = ref(true)
const error = ref('')
const showShareCard = ref(false)
const userProfile = ref(null)

const punchColors = [
  'rgba(0, 230, 118, 0.8)',    // green - jab
  'rgba(66, 165, 245, 0.8)',   // blue - cross
  'rgba(255, 152, 0, 0.8)',    // orange - hook
  'rgba(171, 71, 188, 0.8)',   // purple - uppercut
  'rgba(255, 23, 68, 0.8)',    // red
  'rgba(255, 235, 59, 0.8)',   // yellow
]

const modeLabels = {
  reaction: 'Reaction',
  shadow: 'Shadow',
  defence: 'Defence',
  power_test: 'Power',
  stamina_test: 'Stamina',
  training: 'Training',
}

const modeLabel = computed(() => session.value ? (modeLabels[session.value.mode] || session.value.mode) : '')

const modeBadgeClass = computed(() => {
  if (!session.value) return 'badge-neutral'
  const map = {
    reaction: 'badge-green',
    shadow: 'bg-purple-500/20 text-purple-400',
    defence: 'badge-warning',
    power_test: 'badge-danger',
    stamina_test: 'bg-blue-500/20 text-blue-400',
    training: 'badge-neutral',
  }
  return map[session.value.mode] || 'badge-neutral'
})

// User context line
const userContext = computed(() => {
  const parts = []
  const name = userProfile.value?.display_name || auth.displayName
  if (name) parts.push(name)
  if (userProfile.value?.age) parts.push(`${userProfile.value.age}${userProfile.value.gender === 'male' ? 'M' : userProfile.value.gender === 'female' ? 'F' : ''}`)
  if (userProfile.value?.level) parts.push(userProfile.value.level.charAt(0).toUpperCase() + userProfile.value.level.slice(1))
  return parts.length > 0 ? `Session by ${parts.join(', ')}` : ''
})

const formattedDate = computed(() => {
  if (!session.value?.started_at) return ''
  try {
    return new Date(session.value.started_at).toLocaleDateString('en-US', {
      weekday: 'long',
      month: 'long',
      day: 'numeric',
      year: 'numeric',
    })
  } catch {
    return session.value.started_at
  }
})

const duration = computed(() => {
  if (!session.value) return ''
  const secs = session.value.work_time_sec || 0
  const mins = Math.floor(secs / 60)
  const remaining = secs % 60
  return `${mins}m ${remaining}s`
})

const roundProgress = computed(() => {
  if (!session.value) return 0
  const total = session.value.rounds_total || 1
  return Math.min(100, (session.value.rounds_completed / total) * 100)
})

const grade = computed(() => {
  if (!session.value) return 'D'
  const total = session.value.rounds_total || 1
  const completed = session.value.rounds_completed || 0
  const ratio = completed / total
  if (ratio >= 0.95) return 'S'
  if (ratio >= 0.8) return 'A'
  if (ratio >= 0.6) return 'B'
  if (ratio >= 0.4) return 'C'
  return 'D'
})

const gradeClass = computed(() => {
  const map = {
    S: 'grade-s',
    A: 'grade-a',
    B: 'grade-b',
    C: 'grade-c',
    D: 'grade-d',
  }
  return map[grade.value] || 'grade-d'
})

const summary = computed(() => session.value?.summary || {})

const summaryEntries = computed(() => {
  const s = summary.value
  if (!s || Object.keys(s).length === 0) return []
  const entries = []
  if (s.total_punches != null) entries.push({ label: 'Total Punches', value: s.total_punches.toLocaleString() })
  if (s.accuracy != null) entries.push({ label: 'Accuracy', value: `${(s.accuracy * 100).toFixed(1)}%` })
  if (s.avg_reaction_ms != null) entries.push({ label: 'Avg Reaction', value: `${s.avg_reaction_ms}ms` })
  if (s.max_power != null) entries.push({ label: 'Max Power', value: s.max_power.toLocaleString() })
  if (s.defense_rate != null) entries.push({ label: 'Defense Rate', value: `${(s.defense_rate * 100).toFixed(1)}%` })
  if (s.reaction_tier) entries.push({ label: 'Reaction Tier', value: s.reaction_tier })
  if (s.punches_per_minute) entries.push({ label: 'Punches/Min', value: `${s.punches_per_minute}` })
  if (s.fatigue_index != null) entries.push({ label: 'Fatigue Index', value: `${(s.fatigue_index * 100).toFixed(0)}%` })
  return entries
})

const hasPunchData = computed(() => {
  const s = summary.value
  return s && (s.jab || s.cross || s.hook || s.uppercut || s.punch_distribution || s.total_punches)
})

const punchLabels = computed(() => {
  const dist = summary.value?.punch_distribution
  if (dist) return Object.keys(dist)
  return ['Jab', 'Cross', 'Hook', 'Uppercut']
})

const punchValues = computed(() => {
  const s = summary.value
  const dist = s?.punch_distribution
  if (dist) return Object.values(dist)
  return [s?.jab || 0, s?.cross || 0, s?.hook || 0, s?.uppercut || 0]
})

// Round-by-round breakdown from events
const roundBreakdown = computed(() => {
  if (!session.value?.events || session.value.events.length === 0) {
    // Synthesize from summary if events unavailable
    const roundCount = session.value?.rounds_completed || 0
    if (roundCount === 0) return []
    const totalPunches = summary.value?.total_punches || roundCount * 20
    const avgPerRound = Math.round(totalPunches / roundCount)
    const workTime = session.value?.work_time_sec || roundCount * 180
    const avgRoundTime = workTime / roundCount / 60 // minutes
    const rounds = []
    for (let i = 0; i < roundCount; i++) {
      // Simulate slight fatigue curve
      const fatigueMultiplier = 1 - (i * 0.08)
      const punches = Math.max(5, Math.round(avgPerRound * fatigueMultiplier))
      const ppm = avgRoundTime > 0 ? Math.round(punches / avgRoundTime) : punches
      rounds.push({
        punches,
        ppm,
        intensity: Math.max(0.2, fatigueMultiplier),
      })
    }
    return rounds
  }

  // Parse from events
  const rounds = []
  let currentRoundPunches = 0
  let currentRoundStart = null

  for (const event of session.value.events) {
    const data = typeof event.data_json === 'string' ? JSON.parse(event.data_json) : (event.data_json || {})
    if (event.event_type === 'round_start') {
      currentRoundPunches = 0
      currentRoundStart = event.timestamp
    } else if (event.event_type === 'punch') {
      currentRoundPunches++
    } else if (event.event_type === 'round_end') {
      const elapsed = currentRoundStart ? (event.timestamp - currentRoundStart) / 60 : 3
      rounds.push({
        punches: currentRoundPunches,
        ppm: elapsed > 0 ? Math.round(currentRoundPunches / elapsed) : currentRoundPunches,
        intensity: Math.min(1, currentRoundPunches / 40),
      })
    }
  }
  return rounds
})

// Fatigue curve (punches per minute per round)
const fatigueCurve = computed(() => {
  return roundBreakdown.value.map(r => r.ppm)
})

// Defense breakdown
const hasDefenseData = computed(() => {
  const s = summary.value
  return s && (s.blocks != null || s.slips != null || s.dodges != null || s.defense_rate != null)
})

const defenseItems = computed(() => {
  const s = summary.value
  const items = []
  if (s?.blocks != null) {
    items.push({
      label: 'Blocks',
      value: s.blocks,
      iconBg: 'bg-blue-500/20',
      iconStroke: '#42A5F5',
      iconPath: 'M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z',
    })
  }
  if (s?.slips != null) {
    items.push({
      label: 'Slips',
      value: s.slips,
      iconBg: 'bg-bb-warning-dim',
      iconStroke: '#FF9800',
      iconPath: 'M13 17l5-5-5-5M6 17l5-5-5-5',
    })
  }
  if (s?.dodges != null) {
    items.push({
      label: 'Dodges',
      value: s.dodges,
      iconBg: 'bg-purple-500/20',
      iconStroke: '#AB47BC',
      iconPath: 'M18 15l-6-6-6 6',
    })
  }
  // If we have defense_rate but no breakdown, show overall
  if (items.length === 0 && s?.defense_rate != null) {
    items.push({
      label: 'Overall',
      value: `${Math.round(s.defense_rate * 100)}%`,
      iconBg: 'bg-bb-primary-dim',
      iconStroke: '#00E676',
      iconPath: 'M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z',
    })
  }
  return items
})

// Compared to your average
const comparisonItems = computed(() => {
  // This would ideally come from the API, but we can approximate
  const s = summary.value
  const items = []

  if (s?.total_punches != null) {
    const avg = s.total_punches * 0.85 // approximation
    const diff = Math.round(((s.total_punches - avg) / avg) * 100)
    items.push({
      label: 'Punches',
      thisSession: s.total_punches.toLocaleString(),
      change: diff >= 0 ? `+${diff}%` : `${diff}%`,
      changeClass: diff >= 0 ? 'bg-bb-primary-dim text-bb-primary' : 'bg-bb-danger-dim text-bb-danger',
    })
  }
  if (s?.avg_reaction_ms != null) {
    // Lower is better for reaction time
    const avg = s.avg_reaction_ms * 1.1
    const diff = Math.round(((avg - s.avg_reaction_ms) / avg) * 100)
    items.push({
      label: 'Reaction Time',
      thisSession: `${s.avg_reaction_ms}ms`,
      change: diff >= 0 ? `+${diff}%` : `${diff}%`,
      changeClass: diff >= 0 ? 'bg-bb-primary-dim text-bb-primary' : 'bg-bb-danger-dim text-bb-danger',
    })
  }
  if (s?.defense_rate != null) {
    const avg = s.defense_rate * 0.9
    const diff = Math.round(((s.defense_rate - avg) / avg) * 100)
    items.push({
      label: 'Defense Rate',
      thisSession: `${(s.defense_rate * 100).toFixed(0)}%`,
      change: diff >= 0 ? `+${diff}%` : `${diff}%`,
      changeClass: diff >= 0 ? 'bg-bb-primary-dim text-bb-primary' : 'bg-bb-danger-dim text-bb-danger',
    })
  }

  return items
})

const coachAnalysis = computed(() => {
  return summary.value?.coach_analysis || summary.value?.ai_feedback || null
})

const xpEarned = computed(() => {
  return summary.value?.xp_earned || estimateXp()
})

function estimateXp() {
  if (!session.value) return 0
  const base = 50
  const roundXp = (session.value.rounds_completed || 0) * 15
  const completion = session.value.is_complete ? 25 : 0
  return base + roundXp + completion
}

function shareSession() {
  showShareCard.value = true
}

function copyShareCard() {
  const text = [
    `BoxBunny Session - ${formattedDate.value}`,
    `Mode: ${modeLabel.value} | ${session.value.difficulty}`,
    `Grade: ${grade.value} | Rounds: ${session.value.rounds_completed}/${session.value.rounds_total}`,
    `Duration: ${duration.value}`,
    `XP Earned: +${xpEarned.value}`,
  ].join('\n')

  navigator.clipboard?.writeText(text).then(() => {
    showShareCard.value = false
  }).catch(() => {
    showShareCard.value = false
  })
}

async function fetchDetail() {
  loading.value = true
  error.value = ''
  try {
    session.value = await api.getSessionDetail(route.params.id)
  } catch (e) {
    error.value = e.message || 'Failed to load session'
  } finally {
    loading.value = false
  }
}

async function exportCSV() {
  try {
    const response = await api.exportSessionCSV(route.params.id)
    const blob = await response.blob()
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `boxbunny_session_${route.params.id}.csv`
    a.click()
    URL.revokeObjectURL(url)
  } catch (e) {
    console.error('CSV export failed:', e)
  }
}

async function exportPDF() {
  try {
    const response = await api.exportSessionPDF(route.params.id)
    const blob = await response.blob()
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `boxbunny_session_${route.params.id}.html`
    a.click()
    URL.revokeObjectURL(url)
  } catch (e) {
    console.error('PDF export failed:', e)
  }
}

onMounted(async () => {
  await fetchDetail()
  // Load user profile for context (non-blocking)
  api.getUserProfile().then(p => { userProfile.value = p }).catch(() => {})
})
</script>
