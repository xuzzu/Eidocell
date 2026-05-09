import { ref, watch, computed } from 'vue'

export const STAT_ATTRIBUTES = [
  'area', 'perimeter', 'equivalent_diameter', 'aspect_ratio',
  'solidity', 'form_factor', 'mean_intensity', 'std_intensity',
  'thickness_mean', 'snr',
] as const

export type StatAttribute = typeof STAT_ATTRIBUTES[number]

const STORAGE_KEY = 'eidocell:stats-prefs:visible-attrs'

function load(): Set<string> {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (raw) {
      const parsed = JSON.parse(raw) as string[]
      return new Set(parsed.filter(a => (STAT_ATTRIBUTES as readonly string[]).includes(a)))
    }
  } catch {
    // fall through to default
  }
  return new Set(STAT_ATTRIBUTES)
}

const visibleAttributes = ref<Set<string>>(load())

watch(visibleAttributes, (set) => {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(Array.from(set)))
  } catch {
    // ignore quota errors
  }
}, { deep: true })

export function useStatsPreferences() {
  const visibleList = computed(() =>
    STAT_ATTRIBUTES.filter(a => visibleAttributes.value.has(a)),
  )

  function toggle(attr: string) {
    const next = new Set(visibleAttributes.value)
    if (next.has(attr)) next.delete(attr)
    else next.add(attr)
    visibleAttributes.value = next
  }

  function setAll(visible: boolean) {
    visibleAttributes.value = visible ? new Set(STAT_ATTRIBUTES) : new Set()
  }

  function isVisible(attr: string): boolean {
    return visibleAttributes.value.has(attr)
  }

  return {
    visibleAttributes,
    visibleList,
    toggle,
    setAll,
    isVisible,
  }
}
