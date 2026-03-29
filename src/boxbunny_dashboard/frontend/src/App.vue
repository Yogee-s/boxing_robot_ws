<template>
  <div class="min-h-screen min-h-[100dvh] bg-bb-bg text-bb-text safe-top">
    <router-view v-slot="{ Component, route }">
      <transition name="page" mode="out-in">
        <component :is="Component" :key="route.path" />
      </transition>
    </router-view>
    <NavBar v-if="showNav" />
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import NavBar from '@/components/NavBar.vue'

const route = useRoute()
const auth = useAuthStore()

const showNav = computed(() => {
  const noNavRoutes = ['login', 'chat']
  return auth.isAuthenticated && !noNavRoutes.includes(route.name)
})
</script>
