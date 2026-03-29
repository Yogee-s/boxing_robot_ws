<template>
  <div class="pb-24 px-4 pt-6 max-w-lg mx-auto">
    <!-- Header -->
    <div class="flex items-center justify-between mb-6 animate-fade-in">
      <div>
        <p class="text-bb-text-secondary text-xs font-medium mb-0.5">Welcome back</p>
        <h1 class="text-2xl font-bold text-bb-text">{{ auth.displayName }}</h1>
      </div>
      <div class="flex items-center gap-3">
        <StreakDisplay
          v-if="gamification"
          :streak="gamification.current_streak"
          :longest="gamification.longest_streak"
          :show-label="false"
        />
        <RankBadge
          v-if="gamification"
          :rank="gamification.current_rank"
          :xp="gamification.total_xp"
          size="sm"
          :show-label="false"
        />
      </div>
    </div>

    <!-- User Profile Card -->
    <div v-if="profile" class="card mb-4 animate-slide-up" style="animation-delay: 30ms">
      <div class="flex items-center gap-3 mb-3">
        <div class="w-12 h-12 rounded-xl bg-gradient-to-br from-bb-primary/30 to-bb-primary/5 flex items-center justify-center border border-bb-primary/20">
          <span class="text-bb-primary font-bold text-lg">{{ profileInitial }}</span>
        </div>
        <div class="flex-1 min-w-0">
          <p class="text-sm font-semibold text-bb-text truncate">{{ profile.display_name || auth.displayName }}</p>
          <p class="text-[11px] text-bb-text-muted">
            {{ profileSubtext }}
          </p>
        </div>
        <router-link to="/settings" class="text-bb-text-muted active:opacity-70">
          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"/></svg>
        </router-link>
      </div>
      <div class="grid grid-cols-3 gap-2">
        <div v-for="stat in profileStats" :key="stat.label" class="bg-bb-bg/60 rounded-lg px-2 py-1.5 text-center">
          <p class="text-[10px] text-bb-text-muted uppercase tracking-wide">{{ stat.label }}</p>
          <p class="text-xs font-semibold text-bb-text mt-0.5">{{ stat.value }}</p>
        </div>
      </div>
    </div>

    <!-- XP Progress Bar -->
    <div v-if="gamification" class="card mb-4 animate-slide-up" style="animation-delay: 50ms">
      <div class="flex items-center justify-between mb-2">
        <RankBadge :rank="gamification.current_rank" :xp="gamification.total_xp" size="sm" />
        <span v-if="gamification.next_rank" class="text-xs text-bb-text-muted">
          {{ gamification.xp_to_next_rank.toLocaleString() }} XP to {{ gamification.next_rank }}
        </span>
        <span v-else class="badge-green">Max Rank</span>
      </div>
      <div class="progress-bar">
        <div
          class="progress-fill bg-bb-primary"
          :style="{ width: `${xpProgress}%` }"
        />
      </div>
      <div class="flex justify-between mt-1.5">
        <span class="text-[10px] text-bb-text-muted">{{ gamification.total_xp.toLocaleString() }} XP</span>
        <span v-if="gamification.next_rank" class="text-[10px] text-bb-text-muted">
          {{ nextRankThreshold.toLocaleString() }} XP
        </span>
      </div>
    </div>

    <!-- Weekly Goal + Streak -->
    <div v-if="gamification" class="grid grid-cols-2 gap-3 mb-4">
      <div class="card animate-slide-up" style="animation-delay: 100ms">
        <p class="section-title mb-2">Weekly Goal</p>
        <div class="flex items-baseline gap-1 mb-2">
          <span class="text-xl font-bold text-bb-text">{{ gamification.weekly_progress }}</span>
          <span class="text-sm text-bb-text-secondary">/{{ gamification.weekly_goal }}</span>
          <span class="text-xs text-bb-text-muted ml-1">sessions</span>
        </div>
        <div class="progress-bar">
          <div
            class="progress-fill"
            :class="weeklyGoalMet ? 'bg-bb-primary' : 'bg-bb-warning'"
            :style="{ width: `${weeklyProgress}%` }"
          />
        </div>
      </div>

      <div class="card animate-slide-up" style="animation-delay: 150ms">
        <p class="section-title mb-2">Training Streak</p>
        <StreakDisplay
          :streak="gamification.current_streak"
          :longest="gamification.longest_streak"
        />
      </div>
    </div>

    <!-- Weekly Training Heat Map -->
    <div v-if="trends" class="card mb-4 animate-slide-up" style="animation-delay: 170ms">
      <p class="section-title mb-3">This Week</p>
      <div class="flex items-center justify-between gap-1">
        <div
          v-for="(day, idx) in weekDays"
          :key="idx"
          class="flex flex-col items-center gap-1.5 flex-1"
        >
          <div
            class="w-8 h-8 rounded-lg flex items-center justify-center text-[10px] font-semibold transition-all duration-300"
            :class="trainedOnDay(idx)
              ? 'bg-bb-primary text-bb-bg shadow-sm shadow-bb-primary/30'
              : isToday(idx)
                ? 'bg-bb-surface-lighter text-bb-text-secondary ring-1 ring-bb-border/50'
                : 'bg-bb-surface-light text-bb-text-muted'"
          >
            {{ trainedOnDay(idx) ? '\u2713' : '' }}
          </div>
          <span class="text-[9px] font-medium" :class="isToday(idx) ? 'text-bb-primary' : 'text-bb-text-muted'">
            {{ day }}
          </span>
        </div>
      </div>
      <div class="flex items-center justify-between mt-3 pt-2 border-t border-bb-border/20">
        <span class="text-[10px] text-bb-text-muted">
          {{ trends.weekly_summary?.sessions || 0 }} sessions this week
        </span>
        <span class="text-[10px] text-bb-text-muted">
          {{ (trends.weekly_summary?.total_punches || 0).toLocaleString() }} punches
        </span>
      </div>
    </div>

    <!-- Quick Stats with Trends -->
    <div class="mb-4 animate-slide-up" style="animation-delay: 200ms">
      <h2 class="section-title">Quick Stats</h2>
      <div class="grid grid-cols-2 gap-3">
        <StatCard
          label="Total Sessions"
          :value="sessionStore.totalSessions"
          icon="T"
          color="green"
          :change="trendPctSessions"
          :delay="250"
        />
        <StatCard
          label="Total Punches"
          :value="computedTotalPunches"
          icon="P"
          color="warning"
          :change="trendPctPunches"
          :delay="300"
        />
        <StatCard
          label="Best Defence"
          :value="computedBestDefense"
          unit="%"
          icon="D"
          color="neutral"
          :change="null"
          :delay="350"
        />
        <StatCard
          label="Best Reaction"
          :value="computedBestReaction"
          unit="ms"
          icon="R"
          color="green"
          :change="null"
          :delay="400"
        />
      </div>
    </div>

    <!-- Compared to Peers -->
    <div v-if="benchmarks && Object.keys(benchmarks.benchmarks || {}).length > 0" class="card mb-4 animate-slide-up" style="animation-delay: 250ms">
      <div class="flex items-center justify-between mb-3">
        <h3 class="section-title mb-0">Compared to Peers</h3>
        <span class="text-[10px] text-bb-text-muted">
          {{ peerGroupLabel }}
        </span>
      </div>
      <div class="space-y-3">
        <div v-for="bar in peerBars" :key="bar.label">
          <div class="flex items-center justify-between mb-1">
            <span class="text-xs text-bb-text-secondary">{{ bar.label }}</span>
            <span class="text-[10px] font-semibold" :class="bar.tierColor">{{ bar.tier }}</span>
          </div>
          <div class="progress-bar h-2">
            <div
              class="progress-fill transition-all duration-1000 ease-out"
              :class="bar.barColor"
              :style="{ width: `${bar.percentile}%` }"
            />
          </div>
          <p class="text-[10px] text-bb-text-muted mt-0.5">
            Better than {{ bar.percentile }}% of {{ bar.group }}
          </p>
        </div>
      </div>
    </div>

    <!-- Recent Session -->
    <div class="mb-4 animate-slide-up" style="animation-delay: 300ms">
      <div class="flex items-center justify-between mb-3">
        <h2 class="section-title mb-0">Recent Session</h2>
        <router-link to="/history" class="text-xs text-bb-primary font-medium">
          View All
        </router-link>
      </div>
      <SessionCard v-if="recentSession" :session="recentSession" />
      <div v-else class="card text-center py-8">
        <div class="w-14 h-14 mx-auto mb-3 rounded-2xl bg-bb-surface-light flex items-center justify-center">
          <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" class="text-bb-text-muted"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
        </div>
        <p class="text-bb-text-muted text-sm">No sessions yet</p>
        <p class="text-bb-text-muted text-xs mt-1">Start training to see your data here</p>
      </div>
    </div>

    <!-- AI Coach Says -->
    <div class="card mb-4 animate-slide-up" style="animation-delay: 350ms">
      <div class="flex items-center gap-2 mb-2">
        <div class="w-7 h-7 rounded-lg bg-bb-primary-dim flex items-center justify-center">
          <span class="text-bb-primary text-[10px] font-bold">AI</span>
        </div>
        <h3 class="section-title mb-0">AI Coach Says</h3>
      </div>
      <p class="text-sm text-bb-text-secondary leading-relaxed">{{ coachTip }}</p>
      <router-link
        to="/chat"
        class="inline-flex items-center gap-1 mt-2 text-xs text-bb-primary font-medium"
      >
        Chat with Coach
        <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"/></svg>
      </router-link>
    </div>

    <!-- Quick Actions -->
    <div class="animate-slide-up" style="animation-delay: 400ms">
      <h2 class="section-title">Quick Actions</h2>
      <div class="grid grid-cols-3 gap-3">
        <router-link to="/achievements" class="card-interactive text-center py-4">
          <div class="w-8 h-8 mx-auto mb-1.5 rounded-lg bg-bb-primary-dim flex items-center justify-center">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#FF6B35" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="8" r="7"/><polyline points="8.21 13.89 7 23 12 20 17 23 15.79 13.88"/></svg>
          </div>
          <span class="text-xs text-bb-text-secondary">Achievements</span>
        </router-link>
        <router-link to="/presets" class="card-interactive text-center py-4">
          <div class="w-8 h-8 mx-auto mb-1.5 rounded-lg bg-bb-warning-dim flex items-center justify-center">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#FF9800" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/></svg>
          </div>
          <span class="text-xs text-bb-text-secondary">Presets</span>
        </router-link>
        <router-link
          v-if="auth.isCoach"
          to="/coach"
          class="card-interactive text-center py-4"
        >
          <div class="w-8 h-8 mx-auto mb-1.5 rounded-lg bg-purple-500/20 flex items-center justify-center">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#A855F7" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>
          </div>
          <span class="text-xs text-bb-text-secondary">Coach Mode</span>
        </router-link>
        <router-link
          v-else
          to="/chat"
          class="card-interactive text-center py-4"
        >
          <div class="w-8 h-8 mx-auto mb-1.5 rounded-lg bg-blue-500/20 flex items-center justify-center">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#3B82F6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
          </div>
          <span class="text-xs text-bb-text-secondary">AI Coach</span>
        </router-link>
      </div>
    </div>

    <!-- Loading skeleton -->
    <div v-if="loading" class="space-y-4 mt-8">
      <div class="skeleton h-24 w-full" />
      <div class="grid grid-cols-2 gap-3">
        <div class="skeleton h-20" />
        <div class="skeleton h-20" />
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, ref, onMounted } from 'vue'
import { useAuthStore } from '@/stores/auth'
import { useSessionStore } from '@/stores/session'
import { useWebSocketStore } from '@/stores/websocket'
import * as api from '@/api/client'
import RankBadge from '@/components/RankBadge.vue'
import StreakDisplay from '@/components/StreakDisplay.vue'
import SessionCard from '@/components/SessionCard.vue'
import StatCard from '@/components/StatCard.vue'

const auth = useAuthStore()
const sessionStore = useSessionStore()
const wsStore = useWebSocketStore()

const loading = computed(() => sessionStore.loading)
const gamification = computed(() => sessionStore.gamification)
const recentSession = computed(() => sessionStore.recentSession)

const profile = ref(null)
const benchmarks = ref(null)
const trends = ref(null)

const weekDays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

// XP calculations
const rankThresholds = {
  Novice: 0,
  Contender: 500,
  Fighter: 1500,
  Warrior: 4000,
  Champion: 10000,
  Elite: 25000,
}

const nextRankThreshold = computed(() => {
  if (!gamification.value?.next_rank) return 0
  return rankThresholds[gamification.value.next_rank] || 0
})

const xpProgress = computed(() => {
  if (!gamification.value) return 0
  const currentRank = gamification.value.current_rank
  const currentThreshold = rankThresholds[currentRank] || 0
  const nextThreshold = nextRankThreshold.value
  if (nextThreshold <= currentThreshold) return 100
  const progress = gamification.value.total_xp - currentThreshold
  const range = nextThreshold - currentThreshold
  return Math.min(100, Math.max(0, (progress / range) * 100))
})

const weeklyProgress = computed(() => {
  if (!gamification.value) return 0
  const goal = gamification.value.weekly_goal || 1
  return Math.min(100, (gamification.value.weekly_progress / goal) * 100)
})

const weeklyGoalMet = computed(() => {
  if (!gamification.value) return false
  return gamification.value.weekly_progress >= gamification.value.weekly_goal
})

// Profile computations
const profileInitial = computed(() => {
  const name = profile.value?.display_name || auth.displayName || '?'
  return name.charAt(0).toUpperCase()
})

const profileSubtext = computed(() => {
  if (!profile.value) return ''
  const parts = []
  if (profile.value.age) parts.push(`${profile.value.age}y`)
  if (profile.value.gender) parts.push(profile.value.gender === 'male' ? 'M' : profile.value.gender === 'female' ? 'F' : profile.value.gender)
  if (profile.value.level) parts.push(profile.value.level.charAt(0).toUpperCase() + profile.value.level.slice(1))
  if (profile.value.stance) parts.push(profile.value.stance.charAt(0).toUpperCase() + profile.value.stance.slice(1))
  return parts.join(' / ') || 'Edit your profile in settings'
})

const profileStats = computed(() => {
  if (!profile.value) return []
  const stats = []
  if (profile.value.height_cm) stats.push({ label: 'Height', value: `${profile.value.height_cm}cm` })
  if (profile.value.weight_kg) stats.push({ label: 'Weight', value: `${profile.value.weight_kg}kg` })
  if (profile.value.reach_cm) stats.push({ label: 'Reach', value: `${profile.value.reach_cm}cm` })
  // Fill with defaults if empty
  if (stats.length === 0) {
    stats.push({ label: 'Height', value: '--' })
    stats.push({ label: 'Weight', value: '--' })
    stats.push({ label: 'Reach', value: '--' })
  }
  while (stats.length < 3) {
    if (!profile.value.height_cm) stats.push({ label: 'Height', value: '--' })
    else if (!profile.value.weight_kg) stats.push({ label: 'Weight', value: '--' })
    else stats.push({ label: 'Reach', value: '--' })
  }
  return stats.slice(0, 3)
})

// Trend-aware stats
const computedTotalPunches = computed(() => {
  if (trends.value?.weekly_summary?.total_punches != null) {
    // Sum from all sessions
    const vol = trends.value.punch_volume || []
    return vol.reduce((sum, p) => sum + p.value, 0)
  }
  return sessionStore.history.reduce((sum, s) => sum + (s.rounds_completed || 0) * 20, 0)
})

const computedBestDefense = computed(() => {
  const pb = trends.value?.personal_bests?.best_defense_rate
  if (pb != null) return Math.round(pb * 100)
  return 0
})

const computedBestReaction = computed(() => {
  const pb = trends.value?.personal_bests?.fastest_reaction_ms
  return pb || 0
})

// Parse period_comparison for trend arrows
const trendPctSessions = computed(() => {
  const pc = trends.value?.period_comparison?.vs_last_week
  if (!pc) return null
  const n = parseInt(pc)
  return isNaN(n) ? null : n
})

const trendPctPunches = computed(() => {
  // Approximate from period comparison
  const pc = trends.value?.period_comparison?.vs_last_month
  if (!pc) return null
  const n = parseInt(pc)
  return isNaN(n) ? null : n
})

// Training heat map
function trainedOnDay(dayIdx) {
  if (!trends.value?.training_days) return false
  return trends.value.training_days.includes(dayIdx)
}

function isToday(dayIdx) {
  return new Date().getDay() === (dayIdx + 1) % 7
}

// Peer comparison bars
const peerGroupLabel = computed(() => {
  const d = benchmarks.value?.demographics
  if (!d) return ''
  const parts = []
  if (d.gender) parts.push(d.gender === 'male' ? 'Males' : d.gender === 'female' ? 'Females' : d.gender)
  if (d.age) {
    const decade = Math.floor(d.age / 10) * 10
    parts.push(`${decade}-${decade + 9}`)
  }
  return parts.join(' ') || 'All users'
})

const peerBars = computed(() => {
  const b = benchmarks.value?.benchmarks
  if (!b || Object.keys(b).length === 0) return []

  const bars = []
  const group = peerGroupLabel.value

  if (b.reaction_time != null) {
    const p = Math.min(99, Math.max(1, b.reaction_time.percentile || 50))
    bars.push({
      label: 'Reaction Time',
      percentile: p,
      tier: tierLabel(p),
      tierColor: tierColor(p),
      barColor: 'bg-bb-primary',
      group,
    })
  }
  if (b.punch_rate != null) {
    const p = Math.min(99, Math.max(1, b.punch_rate.percentile || 50))
    bars.push({
      label: 'Punch Rate',
      percentile: p,
      tier: tierLabel(p),
      tierColor: tierColor(p),
      barColor: 'bg-bb-warning',
      group,
    })
  }
  if (b.power != null) {
    const p = Math.min(99, Math.max(1, b.power.percentile || 50))
    bars.push({
      label: 'Power',
      percentile: p,
      tier: tierLabel(p),
      tierColor: tierColor(p),
      barColor: 'bg-bb-danger',
      group,
    })
  }
  if (b.defense != null) {
    const p = Math.min(99, Math.max(1, b.defense.percentile || 50))
    bars.push({
      label: 'Defense',
      percentile: p,
      tier: tierLabel(p),
      tierColor: tierColor(p),
      barColor: 'bg-blue-400',
      group,
    })
  }

  // Fallback if benchmarks exist but fields are differently named
  if (bars.length === 0) {
    for (const [key, val] of Object.entries(b)) {
      if (val && typeof val === 'object' && val.percentile != null) {
        const p = Math.min(99, Math.max(1, val.percentile))
        bars.push({
          label: key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
          percentile: p,
          tier: tierLabel(p),
          tierColor: tierColor(p),
          barColor: 'bg-bb-primary',
          group,
        })
      }
    }
  }

  return bars
})

function tierLabel(percentile) {
  if (percentile >= 90) return 'Elite'
  if (percentile >= 75) return 'Above Average'
  if (percentile >= 50) return 'Average'
  if (percentile >= 25) return 'Below Average'
  return 'Developing'
}

function tierColor(percentile) {
  if (percentile >= 90) return 'text-bb-primary'
  if (percentile >= 75) return 'text-blue-400'
  if (percentile >= 50) return 'text-bb-text-secondary'
  if (percentile >= 25) return 'text-bb-warning'
  return 'text-bb-danger'
}

// AI Coach tip
const coachTips = [
  "Try focusing on your jab consistency this week. A solid jab sets up everything else.",
  "Your defense could use some work. Try the Defence Drill to improve your blocks and slips.",
  "Great streak! Keep the momentum going. Consistency beats intensity for long-term gains.",
  "Consider increasing your round count gradually. Add one extra round each session this week.",
  "Mix up your training modes. Shadow sparring builds different skills than reaction drills.",
  "Recovery is training too. Make sure you're getting enough rest between intense sessions.",
]

const coachTip = computed(() => {
  // Pick a tip based on day of year for variety
  const dayOfYear = Math.floor((Date.now() - new Date(new Date().getFullYear(), 0, 0)) / 86400000)

  // Context-aware tips
  if (!recentSession.value) {
    return "Welcome to BoxBunny! Start your first training session to get personalized coaching insights."
  }

  if (gamification.value && gamification.value.current_streak === 0) {
    return "Time to get back on track! Even a short session keeps your skills sharp. Start a quick reaction drill today."
  }

  if (gamification.value && weeklyGoalMet.value) {
    return "You hit your weekly goal! Outstanding discipline. Consider pushing the intensity up a notch next week."
  }

  return coachTips[dayOfYear % coachTips.length]
})

onMounted(async () => {
  await Promise.all([
    sessionStore.fetchHistory(1, 5),
    sessionStore.fetchGamification(),
    sessionStore.fetchCurrentSession(),
  ])

  // Connect WebSocket
  if (auth.user) {
    wsStore.connect(auth.user.username, auth.user.user_type)
  }

  // Fetch additional data (non-blocking)
  api.getUserProfile().then(p => { profile.value = p }).catch(() => {})
  api.getBenchmarks().then(b => { benchmarks.value = b }).catch(() => {})
  api.getSessionTrends('30d').then(t => { trends.value = t }).catch(() => {})
})
</script>
