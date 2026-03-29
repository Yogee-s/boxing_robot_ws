/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './src/**/*.{vue,js,ts,jsx,tsx}',
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        bb: {
          bg: '#0A0A0A',
          surface: '#141414',
          'surface-light': '#1C1C1C',
          'surface-lighter': '#242424',
          primary: '#FF6B35',
          'primary-dark': '#E55A2B',
          'primary-dim': '#FF6B3520',
          green: '#00E676',
          'green-dark': '#00C853',
          'green-dim': '#00E67620',
          accent: '#FF8A65',
          warning: '#FF9800',
          'warning-dim': '#FF980020',
          danger: '#E53935',
          'danger-dark': '#C62828',
          'danger-dim': '#E5393520',
          text: '#F5F5F5',
          'text-secondary': '#999999',
          'text-muted': '#555555',
          border: '#1E1E1E',
          'border-light': '#2A2A2A',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      animation: {
        'fade-in': 'fadeIn 0.3s ease-out',
        'slide-up': 'slideUp 0.4s ease-out',
        'slide-down': 'slideDown 0.3s ease-out',
        'scale-in': 'scaleIn 0.3s ease-out',
        'pulse-glow': 'pulseGlow 2s ease-in-out infinite',
        'flame': 'flame 1.5s ease-in-out infinite alternate',
        'count-up': 'countUp 0.6s ease-out',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(20px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        slideDown: {
          '0%': { opacity: '0', transform: 'translateY(-10px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        scaleIn: {
          '0%': { opacity: '0', transform: 'scale(0.9)' },
          '100%': { opacity: '1', transform: 'scale(1)' },
        },
        pulseGlow: {
          '0%, 100%': { boxShadow: '0 0 5px rgba(255, 107, 53, 0.3)' },
          '50%': { boxShadow: '0 0 20px rgba(255, 107, 53, 0.6)' },
        },
        flame: {
          '0%': { transform: 'scale(1) rotate(-3deg)', opacity: '0.8' },
          '100%': { transform: 'scale(1.1) rotate(3deg)', opacity: '1' },
        },
        countUp: {
          '0%': { opacity: '0', transform: 'translateY(10px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
      },
    },
  },
  plugins: [
    require('@tailwindcss/forms'),
  ],
}
