<script setup>
import { computed, onMounted, reactive, ref, watch } from 'vue'

const apiBase = import.meta.env.VITE_API_BASE || 'http://localhost:8000/api'

const fileInput = ref(null)
const analysis = ref(null)
const isLoading = ref(false)
const statusMessage = ref('Ready for upload or camera capture')
const preview = ref(null)
const history = ref([])
const activePanel = ref('analyze')
const isLoggedIn = ref(false)
const showAuthModal = ref(false)
const authMode = ref('login')
const authError = ref('')
const currentUser = ref(null)
const historySearch = ref('')
const historyStartDate = ref('')
const historyEndDate = ref('')
const editingRecord = ref(null)
const showDeleteConfirm = ref(null)
const trendPeriod = ref('week')
const showUserMenu = ref(false)
const currentPage = ref('home') // 'home' or 'history'
const activeHistoryTab = ref('history') // 'history' or 'intake'

const authForm = reactive({
  email: '',
  password: '',
  name: '',
})

const manual = reactive({
  dishName: '',
  portion: 150,
  notes: '',
})

const showManualEntry = ref(false)
const isManualLoading = ref(false)
const showMobileSheet = ref(false) // Mobile bottom sheet for manual correction

const profileForm = reactive({
  height_cm: 170,
  weight_kg: 70,
  age: 25,
  gender: 'male',
  activity_level: 'moderate',
})

const tdee = ref(null)
const healthReport = ref(null)

// Dietary intake statistics - current selected period
const intakePeriod = ref('day') // 'day', 'week', 'month'
const currentDate = ref(new Date())

// Month names for display
const monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

// Calculate current date display text
const dateDisplayText = computed(() => {
  const d = currentDate.value
  if (intakePeriod.value === 'day') {
    return `${monthNames[d.getMonth()]} ${d.getDate()}`
  } else if (intakePeriod.value === 'week') {
    const startOfWeek = new Date(d)
    const day = startOfWeek.getDay() || 7
    startOfWeek.setDate(startOfWeek.getDate() - day + 1)
    const endOfWeek = new Date(startOfWeek)
    endOfWeek.setDate(startOfWeek.getDate() + 6)
    return `${monthNames[startOfWeek.getMonth()]} ${startOfWeek.getDate()} - ${monthNames[endOfWeek.getMonth()]} ${endOfWeek.getDate()}`
  } else {
    return `${monthNames[d.getMonth()]} ${d.getFullYear()}`
  }
})

// Calculate intake data based on history records
const intakeStats = computed(() => {
  const records = history.value || []
  let filteredRecords = []
  const now = currentDate.value
  
  if (intakePeriod.value === 'day') {
    // Today's records
    filteredRecords = records.filter(r => {
      const d = new Date(r.created_at)
      return d.toDateString() === now.toDateString()
    })
  } else if (intakePeriod.value === 'week') {
    // This week's records
    const startOfWeek = new Date(now)
    const day = startOfWeek.getDay() || 7
    startOfWeek.setDate(startOfWeek.getDate() - day + 1)
    startOfWeek.setHours(0, 0, 0, 0)
    const endOfWeek = new Date(startOfWeek)
    endOfWeek.setDate(startOfWeek.getDate() + 7)
    filteredRecords = records.filter(r => {
      const d = new Date(r.created_at)
      return d >= startOfWeek && d < endOfWeek
    })
  } else {
    // This month's records
    filteredRecords = records.filter(r => {
      const d = new Date(r.created_at)
      return d.getMonth() === now.getMonth() && d.getFullYear() === now.getFullYear()
    })
  }
  
  const totalCalories = Math.round(filteredRecords.reduce((sum, r) => sum + (r.calories || r.result?.calories || 0), 0))
  const totalCarbs = Math.round(filteredRecords.reduce((sum, r) => sum + (r.carbs || r.result?.carbs || 0), 0) * 10) / 10
  const totalProtein = Math.round(filteredRecords.reduce((sum, r) => sum + (r.protein || r.result?.protein || 0), 0) * 10) / 10
  const totalFat = Math.round(filteredRecords.reduce((sum, r) => sum + (r.fat || r.result?.fat || 0), 0) * 10) / 10
  
  return { totalCalories, totalCarbs, totalProtein, totalFat, records: filteredRecords }
})

// Bar chart data
const barChartData = computed(() => {
  const records = history.value || []
  const now = currentDate.value
  
  if (intakePeriod.value === 'day') {
    // Group by meal: breakfast, lunch, dinner, snack
    const meals = { breakfast: 0, lunch: 0, dinner: 0, snack: 0 }
    const todayRecords = records.filter(r => {
      const d = new Date(r.created_at)
      return d.toDateString() === now.toDateString()
    })
    todayRecords.forEach(r => {
      const d = new Date(r.created_at)
      const hour = d.getHours()
      const minute = d.getMinutes()
      const timeInMinutes = hour * 60 + minute // Convert to minutes for comparison
      const cal = r.calories || r.result?.calories || 0
      
      // Breakfast 05:00 - 10:00 (300 - 600 minutes)
      // Lunch 10:30 - 14:00 (630 - 840 minutes)
      // Dinner 17:00 - 21:00 (1020 - 1260 minutes)
      // Snack: other times
      if (timeInMinutes >= 300 && timeInMinutes < 600) {
        meals.breakfast += cal
      } else if (timeInMinutes >= 630 && timeInMinutes < 840) {
        meals.lunch += cal
      } else if (timeInMinutes >= 1020 && timeInMinutes < 1260) {
        meals.dinner += cal
      } else {
        meals.snack += cal
      }
    })
    return [
      { label: 'Breakfast', value: meals.breakfast },
      { label: 'Lunch', value: meals.lunch },
      { label: 'Dinner', value: meals.dinner },
      { label: 'Snack', value: meals.snack },
    ]
  } else if (intakePeriod.value === 'week') {
    // Group by Monday to Sunday
    const startOfWeek = new Date(now)
    const day = startOfWeek.getDay() || 7
    startOfWeek.setDate(startOfWeek.getDate() - day + 1)
    startOfWeek.setHours(0, 0, 0, 0)
    
    const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    return days.map((label, i) => {
      const dayStart = new Date(startOfWeek)
      dayStart.setDate(startOfWeek.getDate() + i)
      const dayEnd = new Date(dayStart)
      dayEnd.setDate(dayStart.getDate() + 1)
      
      const dayRecords = records.filter(r => {
        const d = new Date(r.created_at)
        return d >= dayStart && d < dayEnd
      })
      const value = dayRecords.reduce((sum, r) => sum + (r.calories || r.result?.calories || 0), 0)
      return { label, value }
    })
  } else {
    // Group by month (display Jan-Dec of current year)
    const year = now.getFullYear()
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    return months.map((label, monthIndex) => {
      const monthRecords = records.filter(r => {
        const d = new Date(r.created_at)
        return d.getMonth() === monthIndex && d.getFullYear() === year
      })
      const value = monthRecords.reduce((sum, r) => sum + (r.calories || r.result?.calories || 0), 0)
      return { label, value }
    })
  }
})

// Bar chart max value
const barChartMax = computed(() => {
  const max = Math.max(...barChartData.value.map(d => d.value), 1)
  // Round up to appropriate scale
  if (max <= 100) return 100
  if (max <= 500) return Math.ceil(max / 100) * 100
  if (max <= 1000) return Math.ceil(max / 200) * 200
  return Math.ceil(max / 500) * 500
})

// Date navigation
function navigateDate(direction) {
  const d = new Date(currentDate.value)
  if (intakePeriod.value === 'day') {
    d.setDate(d.getDate() + direction)
  } else if (intakePeriod.value === 'week') {
    d.setDate(d.getDate() + direction * 7)
  } else {
    d.setMonth(d.getMonth() + direction)
  }
  currentDate.value = d
}

const trend = ref([
  { label: 'Mon', calories: 1800, protein: 90 },
  { label: 'Tue', calories: 1950, protein: 85 },
  { label: 'Wed', calories: 1720, protein: 102 },
  { label: 'Thu', calories: 2100, protein: 98 },
  { label: 'Fri', calories: 1875, protein: 95 },
  { label: 'Sat', calories: 2050, protein: 88 },
  { label: 'Sun', calories: 1780, protein: 92 },
])

const pieSlices = computed(() => {
  const base = analysis.value?.result || {}
  const total = (base.protein || 0) + (base.fat || 0) + (base.carbs || 0) || 1
  
  // Define sector data - using darker colors to be clearly visible on white background
  const rawSlices = [
    { label: 'Protein', value: base.protein || 0, color: '#0891b2' },
    { label: 'Fat', value: base.fat || 0, color: '#f59e0b' },
    { label: 'Carbs', value: base.carbs || 0, color: '#8b5cf6' },
  ]
  
  // Pie chart parameters
  const cx = 200  // Center X
  const cy = 140  // Center Y
  const radius = 90
  const strokeWidth = 32
  const circumference = 2 * Math.PI * radius
  
  // Accumulated angle, starting from 12 o'clock position (-90 degrees)
  let accumulatedAngle = 0
  
  return rawSlices.map((slice, index) => {
    const percent = Math.round((slice.value / total) * 100)
    const sliceAngle = (percent / 100) * 360  // How many degrees this sector occupies
    
    // rotation: SVG circle starts from 3 o'clock by default, subtract 90 to start from 12 o'clock
    const rotation = accumulatedAngle - 90
    
    // Sector center angle (also calculated from 12 o'clock/-90 degrees)
    const midAngle = -90 + accumulatedAngle + sliceAngle / 2
    
    // Calculate arc dasharray
    const arcLength = (sliceAngle / 360) * circumference
    const gapLength = circumference - arcLength
    
    // Update accumulated angle (after calculating current sector)
    accumulatedAngle += sliceAngle
    
    // Calculate guide line position (extending from sector midpoint)
    const midAngleRad = (midAngle * Math.PI) / 180
    
    // Guide line starting point (outer edge of the ring)
    const outerRadius = radius + strokeWidth / 2
    const p1x = cx + outerRadius * Math.cos(midAngleRad)
    const p1y = cy + outerRadius * Math.sin(midAngleRad)
    
    // Guide line turning point (extending outward)
    const extendRadius = radius + 45
    const p2x = cx + extendRadius * Math.cos(midAngleRad)
    const p2y = cy + extendRadius * Math.sin(midAngleRad)
    
    // Determine if sector is on left or right side
    const isRight = Math.cos(midAngleRad) >= 0
    
    // Horizontal line end point
    const lineLength = 30
    const p3x = isRight ? p2x + lineLength : p2x - lineLength
    const p3y = p2y
    
    // Text position
    const textX = isRight ? p3x + 6 : p3x - 6
    const textY = p3y
    
    return {
      ...slice,
      percent,
      arcLength,
      gapLength,
      rotation,
      midAngle,
      p1x, p1y,
      p2x, p2y,
      p3x, p3y,
      textX,
      textY,
      textAnchor: isRight ? 'start' : 'end',
    }
  })
})

const filteredHistory = computed(() => {
  let items = history.value
  if (historySearch.value) {
    const q = historySearch.value.toLowerCase()
    items = items.filter(item => 
      (item.dish_name || '').toLowerCase().includes(q)
    )
  }
  return items
})

// Login gate state
const showLoginGate = ref(false)

// Mobile Bottom Sheet controls
function openMobileSheet() {
  showMobileSheet.value = true
  document.body.style.overflow = 'hidden'
}

function closeMobileSheet() {
  showMobileSheet.value = false
  document.body.style.overflow = ''
}

// Handle Escape key to close sheet
function handleSheetKeydown(e) {
  if (e.key === 'Escape') {
    closeMobileSheet()
  }
}

function normalize(value, expected) {
  return Math.min(1, value / expected || 0)
}

// Handle History entry click
function handleHistoryClick() {
  if (isLoggedIn.value) {
    currentPage.value = 'history'
    activeHistoryTab.value = 'history'
    loadHistory()
  } else {
    showLoginGate.value = true
  }
}

function openFileDialog() {
  fileInput.value?.click()
}

function onFileChange(event) {
  const file = event.target.files?.[0]
  if (!file) return
  preview.value = URL.createObjectURL(file)
  uploadImage(file)
}

async function uploadImage(file) {
  isLoading.value = true
  statusMessage.value = 'Uploading...'
  const formData = new FormData()
  formData.append('file', file)
  const token = localStorage.getItem('accessToken')
  const headers = token ? { Authorization: `Bearer ${token}` } : {}
  try {
    const response = await fetch(`${apiBase}/analyze`, { method: 'POST', body: formData, headers })
    const data = await response.json()
    // Allow unauthenticated users to analyze, but history is not saved
    analysis.value = data
    statusMessage.value = 'Analysis complete'
    if (data.result?.confidence < 0.3 || data.result?.dish_name === 'Unknown Food') {
      statusMessage.value = 'Low confidence - Please verify or enter manually'
      showManualEntry.value = true
    }
    // Only load history for logged-in users
    if (token) {
      await loadHistory()
    }
  } catch (error) {
    console.error(error)
    statusMessage.value = 'Analysis failed - Please enter manually'
    showManualEntry.value = true
    analysis.value = {
      created_at: new Date().toISOString(),
      result: {
        dish_name: 'Unknown',
        calories: 0,
        protein: 0,
        fat: 0,
        carbs: 0,
        portion_grams: 150,
        confidence: 0,
        suggestions: ['Please enter the food details manually'],
        breakdown: {},
      },
    }
  } finally {
    isLoading.value = false
  }
}

async function loadHistory() {
  try {
    const token = localStorage.getItem('accessToken')
    if (!token) {
      history.value = []
      return
    }
    const headers = { Authorization: `Bearer ${token}` }
    let url = `${apiBase}/history?limit=50`
    if (historyStartDate.value) url += `&start_date=${historyStartDate.value}`
    if (historyEndDate.value) url += `&end_date=${historyEndDate.value}`
    const res = await fetch(url, { headers })
    if (res.status === 401) {
      history.value = []
      return
    }
    history.value = await res.json()
  } catch (error) {
    console.error(error)
    history.value = []
  }
}

async function handleAuth() {
  authError.value = ''
  const endpoint = authMode.value === 'login' ? '/auth/login' : '/auth/register'
  const body = authMode.value === 'login' 
    ? { email: authForm.email, password: authForm.password }
    : { email: authForm.email, password: authForm.password, name: authForm.name }
  
  try {
    const res = await fetch(`${apiBase}${endpoint}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
    const data = await res.json()
    if (!res.ok) {
      authError.value = data.detail || 'Authentication failed'
      return
    }
    localStorage.setItem('accessToken', data.access_token)
    localStorage.setItem('refreshToken', data.refresh_token)
    isLoggedIn.value = true
    showAuthModal.value = false
    statusMessage.value = 'Logged in successfully'
    await loadCurrentUser()
    await loadHistory()
  } catch (error) {
    authError.value = 'Network error. Please try again.'
  }
}

async function loadCurrentUser() {
  const token = localStorage.getItem('accessToken')
  if (!token) return
  try {
    const res = await fetch(`${apiBase}/auth/me`, {
      headers: { Authorization: `Bearer ${token}` }
    })
    if (res.ok) {
      currentUser.value = await res.json()
    }
  } catch (e) {
    console.error(e)
  }
}

function logout() {
  localStorage.removeItem('accessToken')
  localStorage.removeItem('refreshToken')
  isLoggedIn.value = false
  currentUser.value = null
  history.value = []
  statusMessage.value = 'Logged out'
}

async function submitManual() {
  if (!manual.dishName.trim()) {
    statusMessage.value = 'Please enter a dish name'
    return
  }
  isManualLoading.value = true
  statusMessage.value = 'Analyzing nutrition for: ' + manual.dishName
  const token = localStorage.getItem('accessToken')
  const headers = { 'Content-Type': 'application/json' }
  if (token) {
    headers['Authorization'] = `Bearer ${token}`
  }
  try {
    const res = await fetch(`${apiBase}/manual`, {
      method: 'POST',
      headers,
      body: JSON.stringify({
        dish_name: manual.dishName.trim(),
        portion_grams: manual.portion || 150,
        notes: manual.notes || null,
      }),
    })
    if (res.ok) {
      const data = await res.json()
      analysis.value = data
      statusMessage.value = `✅ Nutrition info found for "${manual.dishName}"`
      manual.dishName = ''
      manual.notes = ''
      // Only load history if logged in
      if (token) {
        await loadHistory()
      }
    } else {
      statusMessage.value = 'Failed to get nutrition info. Please try again.'
    }
  } catch (error) {
    console.error(error)
    statusMessage.value = 'Error connecting to server'
  } finally {
    isManualLoading.value = false
  }
}

async function deleteHistoryRecord(id) {
  const token = localStorage.getItem('accessToken')
  if (!token) return
  try {
    await fetch(`${apiBase}/history/${id}`, {
      method: 'DELETE',
      headers: { Authorization: `Bearer ${token}` },
    })
    showDeleteConfirm.value = null
    await loadHistory()
  } catch (error) {
    console.error(error)
  }
}

async function saveEditedRecord() {
  if (!editingRecord.value) return
  const token = localStorage.getItem('accessToken')
  try {
    await fetch(`${apiBase}/history/${editingRecord.value.id}`, {
      method: 'PATCH',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({
        dish_name: editingRecord.value.dish_name,
        calories: editingRecord.value.calories,
        protein: editingRecord.value.protein,
        fat: editingRecord.value.fat,
        carbs: editingRecord.value.carbs,
      }),
    })
    editingRecord.value = null
    await loadHistory()
  } catch (error) {
    console.error(error)
  }
}

async function calculateTDEE() {
  try {
    const res = await fetch(`${apiBase}/health/tdee`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(profileForm),
    })
    if (res.ok) {
      tdee.value = await res.json()
    }
  } catch (error) {
    console.error(error)
  }
}

async function loadHealthReport() {
  const token = localStorage.getItem('accessToken')
  if (!token) return
  try {
    const res = await fetch(`${apiBase}/health/report?period=${trendPeriod.value}`, {
      headers: { Authorization: `Bearer ${token}` },
    })
    if (res.ok) {
      healthReport.value = await res.json()
    }
  } catch (error) {
    console.error(error)
  }
}

const gradientStyle = computed(() => {
  const slices = pieSlices.value
  let start = 0
  const stops = slices.map((slice) => {
    const end = start + slice.percent
    const segment = `${slice.color} ${start}% ${end}%`
    start = end
    return segment
  })
  return {
    background: `conic-gradient(${stops.join(',')})`,
  }
})

const goalProgress = computed(() => {
  const target = tdee.value?.tdee || 2000
  const consumed = analysis.value?.result?.calories || 0
  return Math.min(100, (consumed / target) * 100).toFixed(1)
})

watch(trendPeriod, () => {
  if (isLoggedIn.value) loadHealthReport()
})

onMounted(() => {
  const token = localStorage.getItem('accessToken')
  if (token) {
    isLoggedIn.value = true
    loadCurrentUser()
    loadHistory()
    loadHealthReport()
  }
})
</script>

<template>
  <div class="app-container">
    <!-- Top Navigation Bar -->
    <nav class="top-nav">
      <div class="nav-left">
        <span class="logo" @click="currentPage = 'home'">FoodLens</span>
      </div>
      <div class="nav-right">
        <button v-if="!isLoggedIn" class="nav-btn" @click="showAuthModal = true">
          <span class="nav-icon">👤</span>
          <span class="nav-text">Sign In</span>
        </button>
        <div v-else class="user-dropdown">
          <button class="nav-btn user-btn" @click="showUserMenu = !showUserMenu">
            <span class="user-avatar">{{ (currentUser?.name || currentUser?.email || 'U')[0].toUpperCase() }}</span>
          </button>
          <div v-if="showUserMenu" class="dropdown-menu" @click.stop>
            <div class="dropdown-header">
              <p class="dropdown-name">{{ currentUser?.name || 'User' }}</p>
              <p class="dropdown-email">{{ currentUser?.email }}</p>
            </div>
            <div class="dropdown-divider"></div>
            <button class="dropdown-item" @click="currentPage = 'history'; showUserMenu = false; loadHistory()">
              History
            </button>
            <div class="dropdown-divider"></div>
            <button class="dropdown-item danger" @click="logout(); showUserMenu = false">
              Log out
            </button>
          </div>
        </div>
      </div>
    </nav>

    <!-- Click outside to close dropdown -->
    <div v-if="showUserMenu" class="dropdown-backdrop" @click="showUserMenu = false"></div>

    <!-- Main Content -->
    <div class="page" v-if="currentPage === 'home'">
      <header class="hero">
        <div class="hero-content">
          <p class="badge">AI Nutrition · Photo-based Food Recognition</p>
          <h1>FoodLens Smart Diet Assistant</h1>
          <p class="lede">
            Take a photo or upload any food image. Automatically identify dish name, portion, calories, and macronutrients with personalized dietary suggestions.
          </p>
        </div>
        <div class="glass">
          <p class="label">System Status</p>
          <p class="status">{{ statusMessage }}</p>
          <p class="hint">Designed for a smooth, low-friction workflow</p>
          <p v-if="currentUser" class="user-info">Welcome, {{ currentUser.name || currentUser.email }}</p>
        </div>
      </header>

      <input
        ref="fileInput"
        type="file"
        accept="image/*"
        capture="environment"
        class="hidden-input"
        @change="onFileChange"
      />

    <div v-if="showAuthModal" class="modal-overlay" @click.self="showAuthModal = false">
      <div class="modal">
        <h2>{{ authMode === 'login' ? 'Sign In' : 'Create Account' }}</h2>
        <form @submit.prevent="handleAuth">
          <input v-if="authMode === 'register'" v-model="authForm.name" placeholder="Name" required />
          <input v-model="authForm.email" type="email" placeholder="Email" required />
          <input v-model="authForm.password" type="password" placeholder="Password" minlength="8" required />
          <p v-if="authError" class="error">{{ authError }}</p>
          <button class="primary" type="submit">{{ authMode === 'login' ? 'Sign In' : 'Register' }}</button>
        </form>
        <p class="toggle-auth">
          {{ authMode === 'login' ? "Don't have an account?" : 'Already have an account?' }}
          <a href="#" @click.prevent="authMode = authMode === 'login' ? 'register' : 'login'">
            {{ authMode === 'login' ? 'Register' : 'Sign In' }}
          </a>
        </p>
      </div>
    </div>

    <!-- Login Gate Modal -->
    <div v-if="showLoginGate" class="modal-overlay" @click.self="showLoginGate = false">
      <div class="modal login-gate">
        <div class="login-gate-icon">🔐</div>
        <h2>Sign in to Access History</h2>
        <p class="login-gate-desc">
          Log in to save your food analysis results and view your dietary trends over time.
        </p>
        <p class="login-gate-note">
          <span>💡</span> In anonymous mode, your current results won't be saved.
        </p>
        <button class="primary" @click="showLoginGate = false; showAuthModal = true">
          Log in to Save History
        </button>
        <button class="ghost" @click="showLoginGate = false">
          Continue Without Saving
        </button>
      </div>
    </div>

    <main class="main-layout">
      <div class="workspace-container">
        <!-- Left: Main Content (scrollable) -->
        <div class="main-column">
          <!-- Upload Card -->
          <section class="panel capture">
            <div class="panel-header">
              <div>
                <p class="label">Capture/Upload</p>
                <h2>Upload Food Image</h2>
              </div>
            </div>
            <div class="dropzone" @click="openFileDialog">
              <div class="dropzone-icon">📷</div>
              <p class="big">Drop your photo here</p>
              <p class="muted">or click to browse</p>
            </div>
            <div class="preview-section" v-if="preview">
              <p class="label">Preview</p>
              <div class="preview">
                <img :src="preview" alt="preview" />
              </div>
            </div>
            <div class="quick-steps">
              <div class="step"><span class="step-num">1</span> Upload</div>
              <div class="step"><span class="step-num">2</span> Analyze</div>
              <div class="step"><span class="step-num">3</span> Results</div>
            </div>
            
            <!-- History Entry -->
            <div class="history-entry" @click="handleHistoryClick">
              <span>History / My Records</span>
              <span class="history-arrow">›</span>
            </div>

            <!-- Mobile Manual Entry Button (shown when no analysis yet) -->
            <button 
              v-if="!analysis"
              class="mobile-manual-entry-btn"
              @click="openMobileSheet"
              aria-label="Enter dish manually"
            >
              <span>Enter dish name manually</span>
            </button>
          </section>

          <!-- Results Section (only shown when analysis exists) -->
          <section class="panel results" v-if="analysis">
            <div class="panel-header">
              <div>
                <p class="label">Recognition Result</p>
                <h2>{{ analysis.result.dish_name }}</h2>
              </div>
              <div class="badge">Confidence {{ (analysis.result.confidence * 100).toFixed(1) }}%</div>
            </div>
            <div class="grid two">
              <div class="card highlight">
                <p class="label">Calories & Macros</p>
                <h3>{{ analysis.result.calories }} kcal</h3>
                <p class="muted">
                  Protein {{ analysis.result.protein }}g · Fat {{ analysis.result.fat }}g · Carbs {{ analysis.result.carbs }}g
                </p>
                <p class="muted">Estimated portion {{ analysis.result.portion_grams }} g</p>
                <ul class="suggestions">
                  <li v-for="item in analysis.result.suggestions" :key="item">{{ item }}</li>
                </ul>
                <div v-if="analysis.result.alternatives?.length" class="alternatives">
                  <p class="label">Healthier Alternatives</p>
                  <ul>
                    <li v-for="alt in analysis.result.alternatives" :key="alt">{{ alt }}</li>
                  </ul>
                </div>
              </div>
              <div class="card charts">
                <div class="pie-container">
                  <svg class="pie-chart" viewBox="0 0 400 280">
                    <!-- Single layer donut chart - three sectors on same ring (no gaps) -->
                    <g v-for="(slice, index) in pieSlices" :key="slice.label" class="pie-segment">
                      <!-- Sector arc (flat endpoints) -->
                      <circle 
                        cx="200" cy="140" r="90" 
                        fill="none" 
                        :stroke="slice.color" 
                        stroke-width="32"
                        :stroke-dasharray="`${slice.arcLength} ${slice.gapLength}`"
                        :transform="`rotate(${slice.rotation} 200 140)`"
                      />
                      <!-- Label connector line (diagonal + horizontal) -->
                      <polyline 
                        :points="`${slice.p1x},${slice.p1y} ${slice.p2x},${slice.p2y} ${slice.p3x},${slice.p3y}`"
                        fill="none"
                        stroke="rgba(107,114,128,0.4)"
                        stroke-width="1"
                      />
                      <!-- Label text (color matches sector) -->
                      <text 
                        :x="slice.textX" 
                        :y="slice.textY + 4" 
                        class="pie-label"
                        :text-anchor="slice.textAnchor"
                        :fill="slice.color"
                      >{{ slice.label }} {{ slice.percent }}%</text>
                    </g>
                    <!-- Center background circle -->
                    <circle cx="200" cy="140" r="58" fill="#ffffff" />
                    <!-- Center calorie value -->
                    <text x="200" y="136" class="pie-center-value" text-anchor="middle" dominant-baseline="middle">{{ analysis.result.calories }}</text>
                    <text x="200" y="162" class="pie-center-label" text-anchor="middle">kcal</text>
                  </svg>
                </div>
              </div>
            </div>

            <!-- Save Status Banner -->
            <div class="save-status-banner" v-if="!isLoggedIn">
              <span class="save-icon">⚠️</span>
              <span>This result won't be saved. Log in to save history and track trends.</span>
              <button class="login-btn-small" @click="showAuthModal = true">Log in</button>
            </div>
            <div class="save-status-banner saved" v-else>
              <span class="save-icon">✅</span>
              <span>Saved to History</span>
              <button class="view-history-btn" @click="currentPage = 'history'; activeHistoryTab = 'history'; loadHistory()">View History</button>
            </div>

            <!-- Mobile Correction Trigger Button (hidden on desktop) -->
            <button 
              class="mobile-correct-trigger"
              :class="{ 'low-confidence': analysis.result.confidence < 0.5 }"
              @click="openMobileSheet"
              aria-label="Correct dish name"
            >
              <span v-if="analysis.result.confidence < 0.5">⚠️ Low confidence · Correct name</span>
              <span v-else>✏️ Wrong result? Edit</span>
            </button>
          </section>
        </div>

        <!-- Right: Sticky Side Panel -->
        <aside class="side-panel">
          <div class="side-panel-inner">
            <!-- State 1: No analysis yet - Show Manual Entry -->
            <div class="side-panel-content" v-if="!analysis">
              <div class="side-panel-header">
                <div>
                  <p class="label">Manual Entry</p>
                  <h3>Enter Manually</h3>
                </div>
              </div>
              <p class="side-panel-desc">Don't have a photo? Enter the dish name directly.</p>
              <form class="side-form" @submit.prevent="submitManual">
                <input v-model="manual.dishName" placeholder="Dish name (e.g., Caesar Salad)" required />
                <input v-model.number="manual.portion" type="number" placeholder="Portion (g)" min="10" max="2000" />
                <button class="primary" type="submit" :disabled="isManualLoading">
                  {{ isManualLoading ? 'Analyzing...' : 'Get Nutrition' }}
                </button>
              </form>
            </div>

            <!-- State 2: Has analysis with high confidence - Show summary + edit option -->
            <div class="side-panel-content" v-else-if="analysis.result.confidence >= 0.3">
              <div class="side-panel-header">
                <span class="side-panel-icon">✅</span>
                <div>
                  <p class="label">Recognized</p>
                  <h3>{{ analysis.result.dish_name }}</h3>
                </div>
              </div>
              <div class="result-summary">
                <div class="summary-row">
                  <span>Calories</span>
                  <strong>{{ analysis.result.calories }} kcal</strong>
                </div>
                <div class="summary-row">
                  <span>Protein</span>
                  <strong>{{ analysis.result.protein }}g</strong>
                </div>
                <div class="summary-row">
                  <span>Carbs</span>
                  <strong>{{ analysis.result.carbs }}g</strong>
                </div>
                <div class="summary-row">
                  <span>Fat</span>
                  <strong>{{ analysis.result.fat }}g</strong>
                </div>
              </div>
              <div class="side-panel-divider"></div>
              <p class="side-panel-desc">Wrong result? Enter the correct dish name:</p>
              <form class="side-form" @submit.prevent="submitManual">
                <input v-model="manual.dishName" placeholder="Correct dish name..." />
                <button class="ghost" type="submit" :disabled="isManualLoading || !manual.dishName">
                  {{ isManualLoading ? 'Updating...' : 'Override Result' }}
                </button>
              </form>
            </div>

            <!-- State 3: Low confidence - Emphasize manual entry -->
            <div class="side-panel-content low-confidence" v-else>
              <div class="side-panel-header">
                <span class="side-panel-icon warning">⚠️</span>
                <div>
                  <p class="label warning">Low Confidence</p>
                  <h3>Please Verify</h3>
                </div>
              </div>
              <p class="side-panel-desc">AI couldn't identify this dish confidently. Please enter the correct name:</p>
              <form class="side-form" @submit.prevent="submitManual">
                <input v-model="manual.dishName" placeholder="Enter dish name..." required autofocus />
                <input v-model.number="manual.portion" type="number" placeholder="Portion (g)" min="10" max="2000" />
                <button class="primary" type="submit" :disabled="isManualLoading">
                  {{ isManualLoading ? 'Analyzing...' : 'Get Nutrition' }}
                </button>
              </form>
            </div>
          </div>
        </aside>
      </div>
    </main>

    <!-- Mobile Bottom Sheet for Manual Correction -->
    <Teleport to="body">
      <Transition name="sheet">
        <div 
          v-if="showMobileSheet" 
          class="mobile-sheet-overlay"
          @click.self="closeMobileSheet"
          @keydown="handleSheetKeydown"
        >
          <div class="mobile-sheet" role="dialog" aria-modal="true" aria-labelledby="sheet-title">
            <!-- Sheet Handle -->
            <div class="sheet-handle" @click="closeMobileSheet">
              <div class="handle-bar"></div>
            </div>
            
            <!-- Sheet Header -->
            <div class="sheet-header">
              <h3 id="sheet-title">Correct Dish Name</h3>
              <button class="sheet-close" @click="closeMobileSheet" aria-label="Close">
                <span>✕</span>
              </button>
            </div>
            
            <!-- Sheet Content -->
            <div class="sheet-content">
              <p class="sheet-desc">This will update the calories & macros based on the correct dish.</p>
              <form class="sheet-form" @submit.prevent="submitManual(); closeMobileSheet()">
                <input 
                  v-model="manual.dishName" 
                  placeholder="Enter correct dish name..." 
                  required 
                  autofocus
                  class="sheet-input"
                />
                <input 
                  v-model.number="manual.portion" 
                  type="number" 
                  placeholder="Portion (g) - optional" 
                  min="10" 
                  max="2000"
                  class="sheet-input"
                />
                <button 
                  class="primary sheet-submit" 
                  type="submit" 
                  :disabled="isManualLoading || !manual.dishName.trim()"
                >
                  {{ isManualLoading ? 'Updating...' : 'Override Result' }}
                </button>
              </form>
            </div>
          </div>
        </div>
      </Transition>
    </Teleport>

    <div class="floating" v-if="isLoading">Analyzing...</div>
  </div>

  <!-- History Page -->
  <div class="history-page" v-else-if="currentPage === 'history'">
    <aside class="history-sidebar">
      <!-- Back Button -->
      <button class="back-btn" @click="currentPage = 'home'">
        <span>←</span> Back to Home
      </button>
      
      <!-- Main Navigation -->
      <nav class="sidebar-nav">
        <button 
          class="sidebar-item" 
          :class="{ active: activeHistoryTab === 'history' }"
          @click="activeHistoryTab = 'history'"
        >
          <span>Food History</span>
        </button>
        <button 
          class="sidebar-item" 
          :class="{ active: activeHistoryTab === 'intake' }"
          @click="activeHistoryTab = 'intake'"
        >
          <span>Dietary Intake</span>
        </button>
      </nav>
      
      <!-- Spacer -->
      <div class="sidebar-spacer"></div>
    </aside>

    <main class="history-main">
      <!-- History Archive Section -->
      <section class="panel history-section" v-if="activeHistoryTab === 'history'">
        <div class="panel-header">
          <div>
            <p class="label">Food History</p>
            <h2>Sorted by Time</h2>
          </div>
          <div class="badge">GDPR Compliant</div>
        </div>
        <div class="history-filters">
          <input v-model="historySearch" placeholder="Search by dish name..." class="search-input" />
          <input v-model="historyStartDate" type="date" class="date-input" lang="en" />
          <input v-model="historyEndDate" type="date" class="date-input" lang="en" />
          <button class="ghost" @click="loadHistory">Filter</button>
        </div>
        <div class="history-list">
          <div class="history-item" v-for="item in filteredHistory" :key="item.id || item.created_at">
            <div>
              <p class="label">{{ new Date(item.created_at || Date.now()).toLocaleString('en-US') }}</p>
              <h3>{{ item.dish_name || item.result?.dish_name }}</h3>
            </div>
            <div class="history-meta">
              <p class="muted">
                {{ item.calories || item.result?.calories }} kcal · 
                Protein {{ item.protein || item.result?.protein }}g · 
                Carbs {{ item.carbs || item.result?.carbs }}g · 
                Fat {{ item.fat || item.result?.fat }}g · 
                Portion {{ item.portion_grams || item.result?.portion_grams }}g
              </p>
              <div class="history-actions">
                <button class="icon-btn danger" @click="showDeleteConfirm = item.id">🗑️</button>
              </div>
            </div>
            <div v-if="showDeleteConfirm === item.id" class="delete-confirm">
              <p>Are you sure you want to delete this record?</p>
              <button class="primary small" @click="deleteHistoryRecord(item.id)">Yes, Delete</button>
              <button class="ghost small" @click="showDeleteConfirm = null">Cancel</button>
            </div>
          </div>
          <p v-if="!filteredHistory.length" class="muted">No history records</p>
        </div>
      </section>

      <!-- Dietary Intake Statistics -->
      <section class="panel intake-stats-panel" v-if="activeHistoryTab === 'intake'">
        <!-- Period Toggle -->
        <div class="intake-period-toggle">
          <button :class="{ active: intakePeriod === 'day' }" @click="intakePeriod = 'day'">Day</button>
          <button :class="{ active: intakePeriod === 'week' }" @click="intakePeriod = 'week'">Week</button>
          <button :class="{ active: intakePeriod === 'month' }" @click="intakePeriod = 'month'">Month</button>
        </div>
        
        <!-- Date Navigation -->
        <div class="date-navigator">
          <button class="nav-arrow" @click="navigateDate(-1)">‹</button>
          <span class="date-display">{{ dateDisplayText }}</span>
          <button class="nav-arrow" @click="navigateDate(1)">›</button>
        </div>
        
        <!-- Intake Data Summary -->
        <div class="intake-summary">
          <p class="intake-label">Calorie Intake (kcal)</p>
          <h2 class="intake-value">{{ intakeStats.totalCalories }}</h2>
          
          <div class="macro-stats">
            <div class="macro-item">
              <span class="macro-label">Carbs (g)</span>
              <span class="macro-value">{{ intakeStats.totalCarbs }}</span>
            </div>
            <div class="macro-item">
              <span class="macro-label">Protein (g)</span>
              <span class="macro-value">{{ intakeStats.totalProtein }}</span>
            </div>
            <div class="macro-item">
              <span class="macro-label">Fat (g)</span>
              <span class="macro-value">{{ intakeStats.totalFat }}</span>
            </div>
          </div>
        </div>
        
        <!-- Bar Chart -->
        <div class="bar-chart-section">
          <p class="intake-label">Calorie Intake (kcal)</p>
          <div class="bar-chart">
            <!-- Y-axis Scale -->
            <div class="y-axis">
              <span>{{ barChartMax }}</span>
              <span>{{ Math.round(barChartMax * 0.66) }}</span>
              <span>{{ Math.round(barChartMax * 0.33) }}</span>
              <span>0</span>
            </div>
            <!-- Bar Chart Body -->
            <div class="bars-container">
              <div class="grid-lines">
                <div class="grid-line"></div>
                <div class="grid-line"></div>
                <div class="grid-line"></div>
                <div class="grid-line"></div>
              </div>
              <div class="bars">
                <div class="bar-item" v-for="item in barChartData" :key="item.label">
                  <div class="bar" :style="{ height: `${(item.value / barChartMax) * 100}%` }"></div>
                </div>
              </div>
              <div class="bar-labels">
                <span class="bar-label" v-for="item in barChartData" :key="item.label">{{ item.label }}</span>
              </div>
            </div>
          </div>
        </div>
      </section>
    </main>
  </div>
  </div> <!-- end app-container -->
</template>

<style scoped>
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  backdrop-filter: blur(4px);
}

.modal {
  background: #ffffff;
  padding: 28px;
  border-radius: 16px;
  border: 1px solid #e5e7eb;
  width: 90%;
  max-width: 400px;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.15);
}

.modal h2 {
  margin: 0 0 20px;
  color: #111827;
}

.modal form {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.modal input {
  padding: 12px 14px;
  border-radius: 10px;
  border: 1px solid #d1d5db;
  background: #ffffff;
  color: #1f2937;
  font-size: 14px;
}

.modal input:focus {
  outline: none;
  border-color: #10b981;
  box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.1);
}

.error {
  color: #dc2626;
  margin: 0;
  font-size: 14px;
}

.toggle-auth {
  margin-top: 16px;
  text-align: center;
  color: #6b7280;
  font-size: 14px;
}

.toggle-auth a {
  color: #10b981;
  font-weight: 500;
  cursor: pointer;
}

.toggle-auth a:hover {
  text-decoration: underline;
}

.user-info {
  margin-top: 8px;
  color: #059669;
  font-weight: 600;
}

.history-filters {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin: 16px 0;
}

.search-input,
.date-input {
  padding: 10px 14px;
  border-radius: 10px;
  border: 1px solid #d1d5db;
  background: #ffffff;
  color: #1f2937;
  font-size: 14px;
}

.search-input:focus,
.date-input:focus {
  outline: none;
  border-color: #10b981;
  box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.1);
}

.search-input {
  flex: 1;
  min-width: 200px;
}

.date-input {
  width: 160px;
}

.history-meta {
  display: flex;
  align-items: center;
  gap: 12px;
}

.history-actions {
  display: flex;
  gap: 6px;
}

.icon-btn {
  background: #f9fafb;
  border: 1px solid #e5e7eb;
  cursor: pointer;
  font-size: 16px;
  padding: 6px 10px;
  border-radius: 8px;
  transition: all 0.15s ease;
}

.icon-btn:hover {
  background: #f3f4f6;
  border-color: #d1d5db;
}

.icon-btn.danger:hover {
  background: #fef2f2;
  border-color: #fecaca;
}

.edit-form {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  width: 100%;
}

.edit-form input {
  flex: 1;
  min-width: 120px;
  padding: 10px 12px;
  border-radius: 8px;
  border: 1px solid #d1d5db;
  background: #ffffff;
  color: #1f2937;
}

.edit-form input:focus {
  outline: none;
  border-color: #10b981;
}

.edit-actions {
  display: flex;
  gap: 8px;
}

.small {
  padding: 8px 14px;
  font-size: 13px;
}

.delete-confirm {
  margin-top: 12px;
  padding: 12px;
  background: #fef2f2;
  border: 1px solid #fecaca;
  border-radius: 10px;
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  align-items: center;
}

.delete-confirm p {
  margin: 0;
  flex: 1;
  min-width: 200px;
  color: #991b1b;
}

.period-toggle {
  display: flex;
  gap: 4px;
  background: #f3f4f6;
  padding: 4px;
  border-radius: 10px;
}

.period-toggle button {
  padding: 8px 16px;
  border-radius: 8px;
  border: none;
  background: transparent;
  color: #6b7280;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.15s ease;
}

.period-toggle button:hover {
  color: #374151;
}

.period-toggle button.active {
  background: #ffffff;
  color: #111827;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.trend-legend {
  display: flex;
  gap: 20px;
  margin: 12px 0;
}

.dot {
  display: inline-block;
  width: 10px;
  height: 10px;
  border-radius: 50%;
  margin-right: 6px;
}

.tdee-calc {
  margin-top: 20px;
  padding-top: 20px;
  border-top: 1px solid #e5e7eb;
}

.tdee-calc .row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
  margin-bottom: 10px;
}

.tdee-calc input,
.tdee-calc select {
  padding: 10px 12px;
  border-radius: 10px;
  border: 1px solid #d1d5db;
  background: #ffffff;
  color: #1f2937;
  width: 100%;
  margin-bottom: 10px;
  font-size: 14px;
}

.tdee-calc input:focus,
.tdee-calc select:focus {
  outline: none;
  border-color: #10b981;
  box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.1);
}

.tdee-result {
  margin-top: 16px;
  padding: 16px;
  background: #ecfdf5;
  border: 1px solid #a7f3d0;
  border-radius: 12px;
}

.tdee-result p {
  margin: 6px 0;
  color: #065f46;
}

.tdee-result strong {
  color: #047857;
}

.report {
  margin-top: 16px;
}

.report-stats {
  display: flex;
  gap: 32px;
  margin: 16px 0;
}

.stat {
  display: flex;
  flex-direction: column;
  align-items: center;
}

.stat-value {
  font-size: 28px;
  font-weight: 700;
  color: #0891b2;
}

.stat-label {
  font-size: 13px;
  color: #6b7280;
  margin-top: 4px;
}

.recommendations {
  margin-top: 16px;
}

.recommendations ul {
  margin: 10px 0 0;
  padding-left: 20px;
  color: #374151;
}

.recommendations ul li {
  margin-bottom: 8px;
  line-height: 1.5;
}

/* Manual Correction Styles */
.card.manual {
  background: linear-gradient(135deg, #ecfdf5, #ffffff);
  border: 1px solid #a7f3d0;
}

.card.manual .hint {
  font-size: 14px;
  color: #4b5563;
  margin: 8px 0 16px;
  line-height: 1.5;
}

.manual-form {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.manual-form .input-group {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.manual-form .input-group label {
  font-size: 14px;
  font-weight: 500;
  color: #374151;
}

.manual-form input,
.manual-form textarea {
  padding: 12px 14px;
  border-radius: 10px;
  border: 1px solid #d1d5db;
  background: #ffffff;
  color: #1f2937;
  font-size: 15px;
  transition: border-color 0.15s ease, box-shadow 0.15s ease;
}

.manual-form input:focus,
.manual-form textarea:focus {
  outline: none;
  border-color: #10b981;
  box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.1);
}

.manual-form input::placeholder,
.manual-form textarea::placeholder {
  color: #9ca3af;
}

.manual-form textarea {
  min-height: 70px;
  resize: vertical;
}

.manual-form .input-hint {
  font-size: 12px;
  color: #6b7280;
}

.manual-form button[disabled] {
  opacity: 0.6;
  cursor: not-allowed;
}
</style>
