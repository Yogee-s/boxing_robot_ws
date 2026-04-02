<template>
  <nav class="fixed bottom-0 left-0 right-0 z-50 safe-bottom">
    <div class="nav-bar">
      <div class="flex items-stretch justify-around max-w-lg mx-auto">
        <router-link
          v-for="item in navItems"
          :key="item.to"
          :to="item.to"
          class="nav-item"
          :class="{ active: isActive(item.to) }"
        >
          <!-- Active indicator dot -->
          <div
            class="nav-dot"
            :class="isActive(item.to) ? 'opacity-100 scale-100' : 'opacity-0 scale-0'"
          />
          <div class="nav-icon" :class="{ 'text-bb-primary': isActive(item.to) }">
            <component :is="item.icon" />
          </div>
          <span class="nav-label" :class="isActive(item.to) ? 'text-bb-primary' : 'text-bb-text-muted'">
            {{ item.label }}
          </span>
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

const HomeIcon = {
  template: `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>`
}
const HistoryIcon = {
  template: `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>`
}
const ChartIcon = {
  template: `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>`
}
const ChatIcon = {
  template: `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>`
}
const ProfileIcon = {
  template: `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>`
}

const TrainIcon = {
  template: `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"/></svg>`
}

const navItems = computed(() => [
  { to: '/', label: 'Home', icon: HomeIcon },
  { to: '/training', label: 'Train', icon: TrainIcon },
  { to: '/performance', label: 'Stats', icon: ChartIcon },
  { to: '/chat', label: 'Coach', icon: ChatIcon },
  { to: '/settings', label: 'Profile', icon: ProfileIcon },
])

function isActive(path) {
  if (path === '/') return route.path === '/'
  return route.path.startsWith(path)
}
</script>

<style scoped>
.nav-bar {
  background: rgba(10, 10, 10, 0.92);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border-top: 1px solid rgba(255, 255, 255, 0.06);
}
.nav-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  flex: 1;
  position: relative;
  min-height: 64px;
  padding: 10px 4px 14px;
  text-decoration: none;
  -webkit-tap-highlight-color: transparent;
  touch-action: manipulation;
  transition: transform 150ms ease, background-color 150ms ease;
  will-change: transform;
}
.nav-item:active {
  transform: scale(0.88);
  background-color: rgba(255, 107, 53, 0.08);
}
.nav-dot {
  position: absolute;
  top: 4px;
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background-color: #FF6B35;
  transition: opacity 200ms ease, transform 200ms ease;
}
.nav-icon {
  transition: transform 200ms ease, color 200ms ease;
  will-change: transform;
}
.nav-item.active .nav-icon {
  transform: scale(1.12);
}
.nav-label {
  font-size: 11px;
  font-weight: 600;
  margin-top: 4px;
  transition: color 200ms ease;
  letter-spacing: 0.2px;
}
</style>
