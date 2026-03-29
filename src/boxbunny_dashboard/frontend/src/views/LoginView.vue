<template>
  <div class="min-h-screen min-h-[100dvh] flex flex-col items-center justify-center px-6 py-12 bg-bb-bg">
    <!-- Logo -->
    <div class="mb-10 text-center animate-fade-in">
      <div class="w-20 h-20 mx-auto mb-4 rounded-2xl bg-bb-primary-dim flex items-center justify-center glow-primary-sm">
        <span class="text-3xl font-black text-bb-primary">B</span>
      </div>
      <h1 class="text-3xl font-extrabold tracking-tight">
        Box<span class="text-bb-primary">Bunny</span>
      </h1>
      <p class="text-bb-text-secondary text-sm mt-1">AI Boxing Trainer</p>
    </div>

    <!-- Form -->
    <div class="w-full max-w-sm animate-slide-up">
      <!-- Toggle Login / Signup -->
      <div class="flex bg-bb-surface rounded-xl p-1 mb-6">
        <button
          @click="isSignup = false"
          class="flex-1 py-2 rounded-lg text-sm font-semibold transition-all duration-200"
          :class="!isSignup ? 'bg-bb-surface-light text-bb-text' : 'text-bb-text-muted'"
        >
          Log In
        </button>
        <button
          @click="isSignup = true"
          class="flex-1 py-2 rounded-lg text-sm font-semibold transition-all duration-200"
          :class="isSignup ? 'bg-bb-surface-light text-bb-text' : 'text-bb-text-muted'"
        >
          Sign Up
        </button>
      </div>

      <form @submit.prevent="handleSubmit" class="space-y-4">
        <!-- Display name (signup only) -->
        <div v-if="isSignup" class="animate-fade-in">
          <label class="block text-xs font-medium text-bb-text-secondary mb-1.5">
            Display Name
          </label>
          <input
            v-model="displayName"
            type="text"
            placeholder="What should we call you?"
            class="input"
            required
            minlength="1"
            maxlength="128"
          />
        </div>

        <!-- Username -->
        <div>
          <label class="block text-xs font-medium text-bb-text-secondary mb-1.5">
            Username
          </label>
          <input
            v-model="username"
            type="text"
            placeholder="Enter username"
            class="input"
            required
            minlength="3"
            maxlength="64"
            autocapitalize="none"
            autocorrect="off"
          />
        </div>

        <!-- Password -->
        <div>
          <label class="block text-xs font-medium text-bb-text-secondary mb-1.5">
            Password
          </label>
          <input
            v-model="password"
            type="password"
            placeholder="Enter password"
            class="input"
            required
            :minlength="isSignup ? 6 : 1"
          />
        </div>

        <!-- Level (signup only) -->
        <div v-if="isSignup" class="animate-fade-in">
          <label class="block text-xs font-medium text-bb-text-secondary mb-1.5">
            Experience Level
          </label>
          <div class="grid grid-cols-3 gap-2">
            <button
              v-for="lvl in levels"
              :key="lvl.value"
              type="button"
              @click="level = lvl.value"
              class="py-2 px-3 rounded-xl text-xs font-semibold border transition-all duration-200"
              :class="level === lvl.value
                ? 'border-bb-primary bg-bb-primary-dim text-bb-primary'
                : 'border-bb-border/50 bg-bb-surface text-bb-text-secondary'"
            >
              {{ lvl.label }}
            </button>
          </div>
        </div>

        <!-- Error message -->
        <div
          v-if="error"
          class="bg-bb-danger-dim border border-bb-danger/30 rounded-xl px-4 py-3 text-sm text-bb-danger animate-fade-in"
        >
          {{ error }}
        </div>

        <!-- Submit button -->
        <button
          type="submit"
          :disabled="loading"
          class="btn-primary w-full text-base py-3.5 mt-2"
        >
          <svg
            v-if="loading"
            class="animate-spin h-5 w-5"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
          >
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          <span v-else>{{ isSignup ? 'Create Account' : 'Log In' }}</span>
        </button>
      </form>

      <!-- Version info -->
      <p class="text-center text-bb-text-muted text-[10px] mt-8">
        BoxBunny Dashboard v1.0
      </p>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'

const router = useRouter()
const auth = useAuthStore()

const isSignup = ref(false)
const username = ref('')
const password = ref('')
const displayName = ref('')
const level = ref('beginner')
const loading = ref(false)
const error = ref('')

const levels = [
  { value: 'beginner', label: 'Beginner' },
  { value: 'intermediate', label: 'Intermediate' },
  { value: 'advanced', label: 'Advanced' },
]

async function handleSubmit() {
  loading.value = true
  error.value = ''

  try {
    if (isSignup.value) {
      await auth.signup(username.value, password.value, displayName.value, level.value)
    } else {
      await auth.login(username.value, password.value)
    }
    router.push({ name: 'dashboard' })
  } catch (e) {
    error.value = e.message || 'Something went wrong. Please try again.'
  } finally {
    loading.value = false
  }
}
</script>
