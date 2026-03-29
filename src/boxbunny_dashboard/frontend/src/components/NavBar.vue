<template>
  <nav class="fixed bottom-0 left-0 right-0 z-50 safe-bottom">
    <div class="bg-bb-surface/95 backdrop-blur-xl border-t border-bb-border/30">
      <div class="flex items-center justify-around max-w-lg mx-auto px-2 py-1">
        <router-link
          v-for="item in navItems"
          :key="item.to"
          :to="item.to"
          class="nav-item group"
          :class="{ active: isActive(item.to) }"
        >
          <div class="nav-icon" :class="{ 'text-bb-primary': isActive(item.to) }">
            <component :is="item.icon" />
          </div>
          <span
            class="text-[10px] font-medium mt-0.5 transition-colors"
            :class="isActive(item.to) ? 'text-bb-primary' : 'text-bb-text-muted'"
          >
            {{ item.label }}
          </span>
          <div
            v-if="isActive(item.to)"
            class="absolute -top-0.5 left-1/2 -translate-x-1/2 w-5 h-0.5 bg-bb-primary rounded-full"
          />
        </router-link>
      </div>
    </div>
  </nav>
</template>

<script setup>
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import { useAuthStore } from '@/stores/auth'

const route = useRoute()
const auth = useAuthStore()

// SVG icon components rendered inline
const HomeIcon = {
  template: `<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>`
}

const HistoryIcon = {
  template: `<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>`
}

const ChartIcon = {
  template: `<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>`
}

const ChatIcon = {
  template: `<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>`
}

const ProfileIcon = {
  template: `<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>`
}

const navItems = computed(() => {
  const items = [
    { to: '/', label: 'Home', icon: HomeIcon },
    { to: '/history', label: 'History', icon: HistoryIcon },
    { to: '/performance', label: 'Stats', icon: ChartIcon },
    { to: '/chat', label: 'Coach', icon: ChatIcon },
    { to: '/settings', label: 'Profile', icon: ProfileIcon },
  ]
  return items
})

function isActive(path) {
  if (path === '/') return route.path === '/'
  return route.path.startsWith(path)
}
</script>

<style scoped>
.nav-item {
  @apply relative flex flex-col items-center justify-center py-2 px-3 min-w-0
         text-bb-text-muted transition-colors duration-200 no-underline;
}
.nav-item.active {
  @apply text-bb-primary;
}
.nav-icon {
  @apply transition-all duration-200;
}
.nav-item.active .nav-icon {
  @apply transform scale-110;
}
</style>
