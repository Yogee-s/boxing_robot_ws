/**
 * BoxBunny API client.
 *
 * All HTTP communication with the FastAPI backend routes through this module.
 * Bearer tokens are stored in localStorage and attached automatically.
 */

const API_BASE = '/api'

function getToken() {
  return localStorage.getItem('bb_token')
}

function setToken(token) {
  localStorage.setItem('bb_token', token)
}

function clearToken() {
  localStorage.removeItem('bb_token')
}

async function request(method, path, body = null, opts = {}) {
  const headers = {
    'Content-Type': 'application/json',
    ...opts.headers,
  }

  const token = getToken()
  if (token) {
    headers['Authorization'] = `Bearer ${token}`
  }

  const config = {
    method,
    headers,
  }

  if (body !== null && method !== 'GET') {
    config.body = JSON.stringify(body)
  }

  const url = path.startsWith('http') ? path : `${API_BASE}${path}`
  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), opts.timeout || 15000)
  config.signal = controller.signal
  let response
  try {
    response = await fetch(url, config)
  } finally {
    clearTimeout(timeoutId)
  }

  // Handle 204 No Content
  if (response.status === 204) {
    return null
  }

  // Handle non-JSON responses (file downloads)
  const contentType = response.headers.get('content-type') || ''
  if (!contentType.includes('application/json')) {
    if (!response.ok) {
      throw new ApiError(response.status, 'Request failed')
    }
    return response
  }

  const data = await response.json()

  if (!response.ok) {
    throw new ApiError(response.status, data.detail || 'Request failed', data)
  }

  return data
}

class ApiError extends Error {
  constructor(status, message, data = null) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.data = data
  }
}

// ---- Auth ----

export async function login(username, password) {
  const data = await request('POST', '/auth/login', {
    username,
    password,
    device_type: 'phone',
  })
  setToken(data.token)
  return data
}

export async function signup(username, password, displayName, level = 'beginner') {
  const data = await request('POST', '/auth/signup', {
    username,
    password,
    display_name: displayName,
    user_type: 'individual',
    level,
  })
  setToken(data.token)
  return data
}

export async function patternLogin(username, pattern) {
  const data = await request('POST', '/auth/pattern-login', { username, pattern })
  setToken(data.token)
  return data
}

export async function getSession() {
  return request('GET', '/auth/session')
}

export async function logout() {
  try {
    await request('DELETE', '/auth/logout')
  } finally {
    clearToken()
  }
}

export async function updateProfile(updates) {
  return request('PUT', '/auth/profile', updates)
}

export async function setPattern(userId, pattern) {
  return request('POST', '/auth/set-pattern', { user_id: userId, pattern })
}

// ---- Sessions ----

export async function getCurrentSession() {
  return request('GET', '/sessions/current')
}

export async function getSessionHistory(page = 1, pageSize = 20, mode = null) {
  let url = `/sessions/history?page=${page}&page_size=${pageSize}`
  if (mode) url += `&mode=${mode}`
  return request('GET', url)
}

export async function getSessionDetail(sessionId) {
  return request('GET', `/sessions/${sessionId}`)
}

// ---- Gamification ----

export async function getGamificationProfile() {
  return request('GET', '/gamification/profile')
}

export async function getAchievements() {
  return request('GET', '/gamification/achievements')
}

export async function getLeaderboard(coachingSessionId) {
  return request('GET', `/gamification/leaderboard/${coachingSessionId}`)
}

export async function getBenchmarks() {
  return request('GET', '/gamification/benchmarks')
}

// ---- Trends / Analytics ----

export async function getSessionTrends(range = '30d') {
  return request('GET', `/sessions/trends?range=${range}`)
}

export async function getUserProfile() {
  return request('GET', '/auth/session')
}

// ---- Presets ----

export async function getPresets() {
  return request('GET', '/presets/')
}

export async function createPreset(preset) {
  return request('POST', '/presets/', preset)
}

export async function updatePreset(presetId, updates) {
  return request('PUT', `/presets/${presetId}`, updates)
}

export async function deletePreset(presetId) {
  return request('DELETE', `/presets/${presetId}`)
}

export async function togglePresetFavorite(presetId) {
  return request('POST', `/presets/${presetId}/favorite`)
}

// ---- Chat ----

export async function sendChatMessage(message, context = {}) {
  return request('POST', '/chat/message', { message, context })
}

export async function getChatHistory(limit = 50) {
  return request('GET', `/chat/history?limit=${limit}`)
}

// ---- Coach ----

export async function loadCoachConfig(presetId) {
  return request('POST', '/coach/load-config', { preset_id: presetId })
}

export async function startStation(name, presetId = null, config = {}) {
  return request('POST', '/coach/start-station', {
    name,
    preset_id: presetId,
    config,
  })
}

export async function endCoachSession(coachingSessionId) {
  return request('POST', '/coach/end-session', {
    coaching_session_id: coachingSessionId,
  })
}

export async function getLiveParticipants() {
  return request('GET', '/coach/live')
}

export async function getCoachingSessions() {
  return request('GET', '/coach/sessions')
}

// ---- Export ----

export async function exportSessionCSV(sessionId) {
  return request('GET', `/export/session/${sessionId}/csv`)
}

export async function exportSessionPDF(sessionId) {
  return request('GET', `/export/session/${sessionId}/pdf`)
}

export async function exportDateRange(startDate, endDate, mode = null) {
  let url = `/export/range?start_date=${startDate}&end_date=${endDate}`
  if (mode) url += `&mode=${mode}`
  return request('GET', url)
}

// ---- Health ----

export async function healthCheck() {
  return request('GET', '/health')
}

export { getToken, setToken, clearToken, ApiError }
