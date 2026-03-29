<template>
  <div class="pb-24 px-4 pt-6 max-w-lg mx-auto">
    <h1 class="text-2xl font-bold text-bb-text mb-5 animate-fade-in">Achievements</h1>

    <!-- Rank Section -->
    <div v-if="gamification" class="card mb-5 animate-slide-up">
      <div class="flex items-center justify-between mb-4">
        <RankBadge
          :rank="gamification.current_rank"
          :xp="gamification.total_xp"
          size="lg"
          :show-xp="true"
        />
      </div>
      <div v-if="gamification.next_rank">
        <div class="flex justify-between text-xs text-bb-text-muted mb-1.5">
          <span>{{ gamification.current_rank }}</span>
          <span>{{ gamification.next_rank }}</span>
        </div>
        <div class="progress-bar h-2.5">
          <div class="progress-fill bg-bb-green" :style="{ width: `${xpProgress}%` }" />
        </div>
        <p class="text-xs text-bb-text-secondary mt-2 text-center">
          {{ gamification.xp_to_next_rank.toLocaleString() }} XP to {{ gamification.next_rank }}
        </p>
      </div>
      <div v-else class="text-center">
        <span class="badge-green text-sm px-4 py-1">Maximum Rank Achieved</span>
      </div>
    </div>

    <!-- Streak Section -->
    <div v-if="gamification" class="card mb-5 animate-slide-up" style="animation-delay: 50ms">
      <h3 class="section-title">Training Streak</h3>
      <div class="flex items-center justify-center py-3">
        <StreakDisplay
          :streak="gamification.current_streak"
          :longest="gamification.longest_streak"
        />
      </div>
    </div>

    <!-- Achievement Grid -->
    <div class="animate-slide-up" style="animation-delay: 100ms">
      <h3 class="section-title">Badges</h3>

      <!-- Loading -->
      <div v-if="loading" class="grid grid-cols-3 gap-3">
        <div v-for="i in 9" :key="i" class="skeleton h-28 rounded-xl" />
      </div>

      <!-- Grid -->
      <div v-else class="grid grid-cols-3 gap-3">
        <AchievementBadge
          v-for="achievement in allAchievements"
          :key="achievement.id"
          :achievement-id="achievement.id"
          :unlocked="achievement.unlocked"
          :unlocked-at="achievement.unlockedAt"
        />
      </div>
    </div>

    <!-- Stats summary -->
    <div class="card mt-5 animate-slide-up" style="animation-delay: 150ms">
      <div class="flex items-center justify-between">
        <span class="text-sm text-bb-text-secondary">Unlocked</span>
        <span class="text-sm font-bold text-bb-text">
          {{ unlockedCount }} / {{ allAchievements.length }}
        </span>
      </div>
      <div class="progress-bar mt-2">
        <div
          class="progress-fill bg-bb-green"
          :style="{ width: `${(unlockedCount / allAchievements.length) * 100}%` }"
        />
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useSessionStore } from '@/stores/session'
import RankBadge from '@/components/RankBadge.vue'
import StreakDisplay from '@/components/StreakDisplay.vue'
import AchievementBadge from '@/components/AchievementBadge.vue'

const sessionStore = useSessionStore()

const loading = ref(true)
const gamification = computed(() => sessionStore.gamification)

const rankThresholds = {
  Novice: 0,
  Contender: 500,
  Fighter: 1500,
  Warrior: 4000,
  Champion: 10000,
  Elite: 25000,
}

const xpProgress = computed(() => {
  if (!gamification.value?.next_rank) return 100
  const currentThreshold = rankThresholds[gamification.value.current_rank] || 0
  const nextThreshold = rankThresholds[gamification.value.next_rank] || 0
  if (nextThreshold <= currentThreshold) return 100
  const progress = gamification.value.total_xp - currentThreshold
  const range = nextThreshold - currentThreshold
  return Math.min(100, Math.max(0, (progress / range) * 100))
})

const allAchievementIds = [
  'first_blood',
  'century',
  'fury',
  'thousand_fists',
  'speed_demon',
  'weekly_warrior',
  'consistent',
  'iron_chin',
  'marathon',
  'centurion',
  'well_rounded',
  'perfect_round',
]

const allAchievements = computed(() => {
  const unlocked = sessionStore.achievements || []
  const unlockedMap = new Map()
  unlocked.forEach(a => unlockedMap.set(a.achievement_id, a.unlocked_at))

  return allAchievementIds.map(id => ({
    id,
    unlocked: unlockedMap.has(id),
    unlockedAt: unlockedMap.get(id) || '',
  }))
})

const unlockedCount = computed(() => {
  return allAchievements.value.filter(a => a.unlocked).length
})

onMounted(async () => {
  loading.value = true
  await Promise.all([
    sessionStore.fetchGamification(),
    sessionStore.fetchAchievements(),
  ])
  loading.value = false
})
</script>
