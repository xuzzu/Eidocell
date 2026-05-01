import { defineStore } from 'pinia'
import { ref } from 'vue'
import * as settingsApi from '@/api/settings'
import type { AppSettings, AppSettingsUpdate } from '@/types'

export const useSettingsStore = defineStore('settings', () => {
  const settings = ref<AppSettings | null>(null)
  const loading = ref(false)

  async function fetchSettings() {
    loading.value = true
    try {
      settings.value = await settingsApi.getSettings()
    } finally {
      loading.value = false
    }
  }

  async function updateSettings(data: AppSettingsUpdate) {
    settings.value = await settingsApi.updateSettings(data)
  }

  async function resetSettings() {
    settings.value = await settingsApi.resetSettings()
  }

  return { settings, loading, fetchSettings, updateSettings, resetSettings }
})
