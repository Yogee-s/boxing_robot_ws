<template>
  <div class="min-h-screen min-h-[100dvh] flex flex-col items-center justify-center px-6 py-6 pb-20 bg-bb-bg overflow-hidden safe-bottom">
    <!-- Logo -->
    <transition name="logo" appear>
      <div class="mb-4 text-center">
        <div class="w-20 h-20 mx-auto mb-4 rounded-2xl bg-bb-primary-dim flex items-center justify-center glow-primary-sm">
          <span class="text-3xl font-black text-bb-primary">B</span>
        </div>
        <h1 class="text-3xl font-extrabold tracking-tight">
          Box<span class="text-bb-primary">Bunny</span>
        </h1>
        <p class="text-bb-text-secondary text-sm mt-1">AI Boxing Trainer</p>
      </div>
    </transition>

    <!-- ===== AUTH STEP ===== -->
    <transition name="step" mode="out-in">
      <div v-if="step === 'auth'" key="auth" class="w-full max-w-sm">
        <!-- Login / Signup toggle -->
        <div class="flex bg-bb-surface rounded-xl p-1 mb-4 relative overflow-hidden">
          <div
            class="absolute top-1 bottom-1 rounded-lg bg-bb-surface-light transition-all duration-300 ease-out"
            :style="{ left: isSignup ? '50%' : '4px', width: 'calc(50% - 4px)' }"
          />
          <button
            @click="switchMode(false)"
            class="flex-1 py-2 rounded-lg text-sm font-semibold relative z-10 transition-colors duration-200"
            :class="!isSignup ? 'text-bb-text' : 'text-bb-text-muted'"
          >
            Log In
          </button>
          <button
            @click="switchMode(true)"
            class="flex-1 py-2 rounded-lg text-sm font-semibold relative z-10 transition-colors duration-200"
            :class="isSignup ? 'text-bb-text' : 'text-bb-text-muted'"
          >
            Sign Up
          </button>
        </div>

        <!-- Form content with cross-fade -->
        <transition name="form-switch" mode="out-in">
          <form v-if="!isSignup" key="login-form" @submit.prevent="handleLogin" class="space-y-3">
            <!-- Login method toggle -->
            <div class="flex bg-bb-bg rounded-xl p-1 relative overflow-hidden">
              <div
                class="absolute top-1 bottom-1 rounded-lg bg-bb-surface-light transition-all duration-300 ease-out"
                :style="{ left: loginWithPattern ? '50%' : '4px', width: 'calc(50% - 4px)' }"
              />
              <button
                type="button" @click="loginWithPattern = false"
                class="flex-1 py-1.5 rounded-lg text-xs font-semibold relative z-10 transition-colors duration-200"
                :class="!loginWithPattern ? 'text-bb-text' : 'text-bb-text-muted'"
              >
                Password
              </button>
              <button
                type="button" @click="loginWithPattern = true"
                class="flex-1 py-1.5 rounded-lg text-xs font-semibold relative z-10 transition-colors duration-200"
                :class="loginWithPattern ? 'text-bb-text' : 'text-bb-text-muted'"
              >
                Pattern
              </button>
            </div>

            <!-- Username with account dropdown -->
            <div class="relative">
              <label class="block text-xs font-medium text-bb-text-secondary mb-1.5">Username</label>
              <input
                v-model="username" type="text" placeholder="Enter or select username" class="input"
                required autocapitalize="none" autocorrect="off"
                @focus="showDropdown = true"
                @input="showDropdown = true"
              />
              <!-- Dropdown -->
              <transition name="dropdown">
                <div
                  v-if="showDropdown && filteredAccounts.length > 0"
                  class="absolute z-20 left-0 right-0 top-full mt-1 bg-bb-surface border border-bb-border/50 rounded-xl overflow-hidden shadow-lg shadow-black/40 max-h-48 overflow-y-auto"
                >
                  <button
                    v-for="acc in filteredAccounts" :key="acc.id"
                    @mousedown.prevent="pickAccount(acc)"
                    class="w-full flex items-center gap-3 px-4 py-3 text-left transition-colors duration-100 active:bg-bb-primary-dim"
                    :class="username === acc.username ? 'bg-bb-surface-light' : 'hover:bg-bb-surface-light'"
                  >
                    <div class="w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold"
                      :class="username === acc.username ? 'bg-bb-primary text-white' : 'bg-bb-surface-lighter text-bb-text-secondary'"
                    >{{ acc.display_name.charAt(0).toUpperCase() }}</div>
                    <div class="flex-1 min-w-0">
                      <p class="text-sm font-semibold text-bb-text truncate">{{ acc.display_name }}</p>
                      <p class="text-[11px] text-bb-text-muted">{{ acc.level }}</p>
                    </div>
                    <span v-if="acc.has_pattern" class="text-[10px] text-bb-primary-dim bg-bb-primary/10 px-2 py-0.5 rounded-full">Pattern</span>
                  </button>
                </div>
              </transition>
            </div>

            <!-- Password login -->
            <transition name="field" mode="out-in">
              <div v-if="!loginWithPattern" key="login-pw">
                <label class="block text-xs font-medium text-bb-text-secondary mb-1.5">Password</label>
                <input v-model="password" type="password" placeholder="Enter password" class="input" required />
              </div>
              <div v-else key="login-pattern">
                <label class="block text-xs font-medium text-bb-text-secondary mb-2">Draw Your Pattern</label>
                <div class="flex justify-center">
                  <PatternLock
                    ref="loginPatternRef"
                    :size="200"
                    :error="!!error"
                    @update:pattern="loginPattern = $event"
                    @complete="loginPattern = $event"
                  />
                </div>
              </div>
            </transition>

            <!-- Error -->
            <transition name="field">
              <div
                v-if="error" key="error"
                class="bg-bb-danger-dim border border-bb-danger/30 rounded-xl px-4 py-3 text-sm text-bb-danger"
              >
                {{ error }}
              </div>
            </transition>

            <!-- Submit -->
            <button type="submit" :disabled="loading || !canLoginSubmit" class="btn-primary w-full text-base py-3.5">
              <svg v-if="loading" class="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              <span v-else>Log In</span>
            </button>
          </form>

          <form v-else key="signup-form" @submit.prevent="handleSignup" class="space-y-3">
            <!-- Display name -->
            <div>
              <label class="block text-xs font-medium text-bb-text-secondary mb-1.5">Display Name</label>
              <input
                v-model="displayName" type="text" placeholder="What should we call you?"
                class="input" required minlength="1" maxlength="128"
              />
            </div>

            <!-- Username -->
            <div>
              <label class="block text-xs font-medium text-bb-text-secondary mb-1.5">Username</label>
              <input
                v-model="username" type="text" placeholder="Choose a username" class="input"
                required minlength="3" maxlength="64" autocapitalize="none" autocorrect="off"
              />
            </div>

            <!-- Security method toggle -->
            <div>
              <label class="block text-xs font-medium text-bb-text-secondary mb-2">Security Method</label>
              <div class="flex bg-bb-bg rounded-xl p-1 relative overflow-hidden">
                <div
                  class="absolute top-1 bottom-1 rounded-lg bg-bb-surface-light transition-all duration-300 ease-out"
                  :style="{ left: usePassword ? '50%' : '4px', width: 'calc(50% - 4px)' }"
                />
                <button
                  type="button" @click="usePassword = false"
                  class="flex-1 py-1.5 rounded-lg text-xs font-semibold relative z-10 transition-colors duration-200"
                  :class="!usePassword ? 'text-bb-text' : 'text-bb-text-muted'"
                >
                  Pattern Lock
                </button>
                <button
                  type="button" @click="usePassword = true"
                  class="flex-1 py-1.5 rounded-lg text-xs font-semibold relative z-10 transition-colors duration-200"
                  :class="usePassword ? 'text-bb-text' : 'text-bb-text-muted'"
                >
                  Password
                </button>
              </div>
            </div>

            <!-- Password or Pattern -->
            <transition name="field" mode="out-in">
              <div v-if="usePassword" key="signup-pw">
                <label class="block text-xs font-medium text-bb-text-secondary mb-1.5">Password</label>
                <input v-model="password" type="password" placeholder="Min 6 characters" class="input" required minlength="6" />
              </div>
              <div v-else key="signup-pattern">
                <label class="block text-xs font-medium text-bb-text-secondary mb-2">Draw Your Pattern</label>
                <div class="flex justify-center">
                  <PatternLock
                    ref="signupPatternRef"
                    :size="200"
                    @update:pattern="signupPattern = $event"
                    @complete="signupPattern = $event"
                  />
                </div>
              </div>
            </transition>

            <!-- Error -->
            <transition name="field">
              <div
                v-if="error" key="signup-error"
                class="bg-bb-danger-dim border border-bb-danger/30 rounded-xl px-4 py-3 text-sm text-bb-danger"
              >
                {{ error }}
              </div>
            </transition>

            <!-- Submit -->
            <button type="submit" :disabled="loading || !canSignupSubmit" class="btn-primary w-full text-base py-3.5">
              <svg v-if="loading" class="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              <span v-else>Create Account</span>
            </button>
          </form>
        </transition>

        <p class="text-center text-bb-text-muted text-[10px] mt-8">BoxBunny Dashboard v1.0</p>
      </div>

      <!-- ===== SURVEY STEP ===== -->
      <div v-else-if="step === 'survey'" key="survey" class="w-full max-w-sm">
        <div class="text-center mb-6">
          <p class="text-bb-text-secondary text-sm">Let's find your level</p>
          <h2 class="text-xl font-bold text-bb-text mt-1">Quick Assessment</h2>
          <div class="mt-4 progress-bar h-1.5">
            <div class="progress-fill bg-bb-primary" :style="{ width: `${((currentQuestion + 1) / surveyQuestions.length) * 100}%` }" />
          </div>
          <p class="text-[11px] text-bb-text-muted mt-2">{{ currentQuestion + 1 }} of {{ surveyQuestions.length }}</p>
        </div>

        <transition :name="slideDir" mode="out-in">
          <div :key="currentQuestion" class="card">
            <p class="text-sm font-semibold text-bb-text mb-4 leading-relaxed">{{ surveyQuestions[currentQuestion].question }}</p>
            <div class="space-y-2">
              <button
                v-for="(opt, idx) in surveyQuestions[currentQuestion].options"
                :key="idx" @click="selectAnswer(idx)"
                class="w-full py-3 px-4 rounded-xl text-sm font-medium text-left transition-all duration-200 active:scale-[0.98]"
                :class="surveyAnswers[currentQuestion] === idx
                  ? 'bg-bb-primary text-bb-bg ring-2 ring-bb-primary shadow-sm shadow-bb-primary/20'
                  : 'bg-bb-surface-light text-bb-text-secondary border border-bb-border/30'"
              >
                {{ opt }}
              </button>
            </div>
          </div>
        </transition>

        <div class="flex gap-3 mt-4">
          <button v-if="currentQuestion > 0" @click="prevQuestion" class="btn-secondary flex-1">Back</button>
          <button @click="nextQuestion" :disabled="surveyAnswers[currentQuestion] === null" class="btn-primary flex-1">
            {{ currentQuestion === surveyQuestions.length - 1 ? 'See Results' : 'Next' }}
          </button>
        </div>
      </div>

      <!-- ===== RESULT STEP ===== -->
      <div v-else-if="step === 'result'" key="result" class="w-full max-w-sm text-center">
        <div class="card py-8 mb-4">
          <div class="w-16 h-16 mx-auto mb-4 rounded-2xl bg-bb-primary-dim flex items-center justify-center">
            <span class="text-2xl">{{ suggestedLevel === 'beginner' ? '🥊' : suggestedLevel === 'intermediate' ? '🔥' : '⚡' }}</span>
          </div>
          <h2 class="text-xl font-bold text-bb-text mb-1 capitalize">{{ suggestedLevel }}</h2>
          <p class="text-sm text-bb-text-secondary leading-relaxed max-w-xs mx-auto">{{ levelDescription }}</p>
        </div>
        <p class="text-xs text-bb-text-muted mb-3">Or choose a different level:</p>
        <div class="grid grid-cols-3 gap-2 mb-6">
          <button
            v-for="lvl in levels" :key="lvl.value" @click="suggestedLevel = lvl.value"
            class="py-2.5 px-3 rounded-xl text-xs font-semibold border transition-all duration-200 active:scale-95"
            :class="suggestedLevel === lvl.value
              ? 'border-bb-primary bg-bb-primary-dim text-bb-primary'
              : 'border-bb-border/50 bg-bb-surface text-bb-text-secondary'"
          >
            {{ lvl.label }}
          </button>
        </div>
        <button @click="finishSurvey" :disabled="loading" class="btn-primary w-full text-base py-3.5">
          <svg v-if="loading" class="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          <span v-else>Let's Go</span>
        </button>
      </div>
    </transition>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import * as api from '@/api/client'
import PatternLock from '@/components/PatternLock.vue'

const router = useRouter()
const auth = useAuthStore()

// ---- State ----
const step = ref('auth')
const isSignup = ref(false)
const username = ref('')
const password = ref('')
const displayName = ref('')
const usePassword = ref(false)
const loginWithPattern = ref(false)
const signupPattern = ref([])
const loginPattern = ref([])
const signupPatternRef = ref(null)
const loginPatternRef = ref(null)
const loading = ref(false)
const error = ref('')
const slideDir = ref('slide-left')
const newUserId = ref(null)
const accounts = ref([])
const showDropdown = ref(false)

const filteredAccounts = computed(() => {
  const q = username.value.toLowerCase().trim()
  if (!q) return accounts.value
  return accounts.value.filter(a =>
    a.username.toLowerCase().includes(q) ||
    a.display_name.toLowerCase().includes(q)
  )
})

// Load account list on mount
;(async () => {
  try {
    accounts.value = await api.listUsers()
  } catch {
    accounts.value = []
  }
})()

function pickAccount(acc) {
  username.value = acc.username
  showDropdown.value = false
  if (acc.has_pattern) {
    loginWithPattern.value = true
  }
  error.value = ''
}

function switchMode(signup) {
  error.value = ''
  isSignup.value = signup
}

const canLoginSubmit = computed(() => {
  if (!username.value) return false
  if (loginWithPattern.value) return loginPattern.value.length >= 4
  return !!password.value
})

const canSignupSubmit = computed(() => {
  if (!displayName.value || !username.value) return false
  if (usePassword.value) return password.value.length >= 6
  return signupPattern.value.length >= 4
})

// ---- Login ----
async function handleLogin() {
  loading.value = true
  error.value = ''
  try {
    if (loginWithPattern.value) {
      const data = await api.patternLogin(username.value, loginPattern.value)
      auth.token = data.token
      auth.user = {
        user_id: data.user_id,
        username: data.username,
        display_name: data.display_name,
        user_type: data.user_type,
      }
    } else {
      await auth.login(username.value, password.value)
    }
    router.push({ name: 'dashboard' })
  } catch (e) {
    error.value = e.message || 'Invalid credentials'
    if (loginPatternRef.value) loginPatternRef.value.reset()
  } finally {
    loading.value = false
  }
}

// ---- Signup ----
async function handleSignup() {
  loading.value = true
  error.value = ''
  try {
    const pwd = usePassword.value ? password.value : crypto.randomUUID().slice(0, 16)
    const data = await auth.signup(username.value, pwd, displayName.value, 'beginner')
    newUserId.value = data.user_id
    if (!usePassword.value && signupPattern.value.length >= 4) {
      await api.setPattern(data.user_id, signupPattern.value)
    }
    step.value = 'survey'
  } catch (e) {
    error.value = e.message || 'Something went wrong'
  } finally {
    loading.value = false
  }
}

// ---- Survey ----
const surveyQuestions = [
  { question: 'Have you trained boxing before?', options: ['Never', 'A few times', 'Regularly'] },
  { question: 'Do you know the basic punches (jab, cross, hook, uppercut)?', options: ['No', 'Somewhat', 'Yes'] },
  { question: 'Can you throw a basic 1-2-3 combo?', options: ['No', 'With help', 'Yes'] },
  { question: 'Have you done any sparring before?', options: ['Never', 'Once or twice', 'Yes, regularly'] },
  { question: 'How would you describe your fitness level?', options: ['Low', 'Moderate', 'High'] },
  { question: 'Have you used boxing equipment before (bag, pads)?', options: ['Never', 'Occasionally', 'Often'] },
]

const currentQuestion = ref(0)
const surveyAnswers = ref(Array(surveyQuestions.length).fill(null))
const suggestedLevel = ref('beginner')
const levels = [
  { value: 'beginner', label: 'Beginner' },
  { value: 'intermediate', label: 'Intermediate' },
  { value: 'advanced', label: 'Advanced' },
]
const levelDescription = computed(() => ({
  beginner: "New to boxing. You'll start with fundamental punches and basic combos.",
  intermediate: "Some boxing experience. You'll work on combinations and technique.",
  advanced: "Experienced boxer. You'll tackle complex combos and sparring modes.",
}[suggestedLevel.value]))

function selectAnswer(idx) { surveyAnswers.value[currentQuestion.value] = idx }

function nextQuestion() {
  if (surveyAnswers.value[currentQuestion.value] === null) return
  if (currentQuestion.value < surveyQuestions.length - 1) {
    slideDir.value = 'slide-left'
    currentQuestion.value++
  } else {
    const score = surveyAnswers.value.reduce((s, v) => s + (v || 0), 0)
    suggestedLevel.value = score <= 4 ? 'beginner' : score <= 8 ? 'intermediate' : 'advanced'
    step.value = 'result'
  }
}

function prevQuestion() {
  if (currentQuestion.value > 0) {
    slideDir.value = 'slide-right'
    currentQuestion.value--
  }
}

async function finishSurvey() {
  loading.value = true
  try {
    await api.updateProfile({
      level: suggestedLevel.value,
      proficiency_answers_json: JSON.stringify({
        answers: surveyAnswers.value,
        score: surveyAnswers.value.reduce((s, v) => s + (v || 0), 0),
        suggested_level: suggestedLevel.value,
      }),
    })
  } catch { /* proceed anyway */ }
  loading.value = false
  router.push({ name: 'dashboard' })
}
</script>

<style scoped>
/* Logo entrance */
.logo-enter-active { transition: all 0.6s cubic-bezier(0.16, 1, 0.3, 1); }
.logo-enter-from { opacity: 0; transform: translateY(-20px) scale(0.95); }

/* Step transitions (auth <-> survey <-> result) */
.step-enter-active { transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1); }
.step-leave-active { transition: all 0.25s ease-in; }
.step-enter-from { opacity: 0; transform: translateY(20px); }
.step-leave-to { opacity: 0; transform: translateY(-10px); }

/* Form switch (login <-> signup) */
.form-switch-enter-active { transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1); }
.form-switch-leave-active { transition: all 0.2s ease-in; }
.form-switch-enter-from { opacity: 0; transform: translateX(16px); }
.form-switch-leave-to { opacity: 0; transform: translateX(-16px); }

/* Field show/hide */
.field-enter-active { transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1); }
.field-leave-active { transition: all 0.2s ease-in; }
.field-enter-from { opacity: 0; transform: translateY(-6px); }
.field-leave-to { opacity: 0; transform: translateY(-6px); }

/* Survey slide */
.slide-left-enter-active, .slide-left-leave-active,
.slide-right-enter-active, .slide-right-leave-active {
  transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
}
.slide-left-enter-from { opacity: 0; transform: translateX(40px); }
.slide-left-leave-to { opacity: 0; transform: translateX(-40px); }
.slide-right-enter-from { opacity: 0; transform: translateX(-40px); }
.slide-right-leave-to { opacity: 0; transform: translateX(40px); }

/* Dropdown */
.dropdown-enter-active { transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1); }
.dropdown-leave-active { transition: all 0.15s ease-in; }
.dropdown-enter-from { opacity: 0; transform: translateY(-8px) scale(0.97); }
.dropdown-leave-to { opacity: 0; transform: translateY(-8px) scale(0.97); }
</style>
