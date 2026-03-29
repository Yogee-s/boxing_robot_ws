<template>
  <div class="pb-24 px-4 pt-6 max-w-lg mx-auto">
    <!-- Header -->
    <h1 class="text-2xl font-bold text-bb-text mb-5 animate-fade-in">Session History</h1>

    <!-- Filter Tabs -->
    <div class="flex gap-2 mb-5 overflow-x-auto pb-1 scrollbar-hide animate-slide-up">
      <button
        v-for="tab in tabs"
        :key="tab.value"
        @click="activeTab = tab.value"
        class="flex-shrink-0 px-4 py-2 rounded-xl text-xs font-semibold transition-all duration-200"
        :class="activeTab === tab.value
          ? 'bg-bb-primary text-bb-bg'
          : 'bg-bb-surface text-bb-text-secondary border border-bb-border/30'"
      >
        {{ tab.label }}
      </button>
    </div>

    <!-- Loading -->
    <div v-if="loading && sessions.length === 0" class="space-y-3">
      <div v-for="i in 5" :key="i" class="skeleton h-20 w-full rounded-2xl" />
    </div>

    <!-- Empty state -->
    <div v-else-if="sessions.length === 0" class="card text-center py-16">
      <div class="text-4xl mb-3 text-bb-text-muted">--</div>
      <p class="text-bb-text-muted text-sm">No sessions found</p>
      <p class="text-bb-text-muted text-xs mt-1">
        {{ activeTab === 'all' ? 'Start training to see your history' : `No ${activeTab} sessions yet` }}
      </p>
    </div>

    <!-- Session list -->
    <div v-else class="space-y-3">
      <SessionCard
        v-for="(session, idx) in sessions"
        :key="session.session_id"
        :session="session"
        :delay="idx * 50"
      />
    </div>

    <!-- Load more -->
    <div v-if="hasMore" class="mt-4 text-center">
      <button
        @click="loadMore"
        :disabled="loading"
        class="btn-secondary text-sm"
      >
        <svg v-if="loading" class="animate-spin h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
          <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
        </svg>
        <span v-else>Load More</span>
      </button>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted } from 'vue'
import { useSessionStore } from '@/stores/session'
import SessionCard from '@/components/SessionCard.vue'

const sessionStore = useSessionStore()

const activeTab = ref('all')
const page = ref(1)
const pageSize = 20

const tabs = [
  { label: 'All', value: 'all' },
  { label: 'Reaction', value: 'reaction' },
  { label: 'Shadow', value: 'shadow' },
  { label: 'Defence', value: 'defence' },
  { label: 'Power', value: 'power_test' },
]

const loading = computed(() => sessionStore.loading)
const sessions = computed(() => sessionStore.history)
const hasMore = computed(() => {
  return sessionStore.historyTotal > sessions.value.length
})

async function fetchSessions() {
  const mode = activeTab.value === 'all' ? null : activeTab.value
  await sessionStore.fetchHistory(page.value, pageSize, mode)
}

async function loadMore() {
  page.value++
  await fetchSessions()
}

watch(activeTab, () => {
  page.value = 1
  fetchSessions()
})

onMounted(fetchSessions)
</script>

<style scoped>
.scrollbar-hide::-webkit-scrollbar {
  display: none;
}
.scrollbar-hide {
  -ms-overflow-style: none;
  scrollbar-width: none;
}
</style>
