import { watch } from 'vue'
import { useSessionStore } from '@/stores/session'
import { useGalleryStore } from '@/stores/gallery'
import { useClustersStore } from '@/stores/clusters'
import { useClassesStore } from '@/stores/classes'
import { useSegmentationStore } from '@/stores/segmentation'
import { useAnalysisStore } from '@/stores/analysis'
import { useLearningStore } from '@/stores/learning'

/**
 * Watches for session changes and resets all workspace stores.
 * Call once in App.vue to ensure clean state on session switch.
 */
export function useSessionWatcher() {
  const session = useSessionStore()
  const gallery = useGalleryStore()
  const clusters = useClustersStore()
  const classes = useClassesStore()
  const segmentation = useSegmentationStore()
  const analysis = useAnalysisStore()
  const learning = useLearningStore()

  watch(
    () => session.currentSessionId,
    (_newId, oldId) => {
      // Only reset if switching from one session to another (not initial load)
      if (oldId !== undefined) {
        gallery.$reset()
        clusters.$reset()
        classes.$reset()
        segmentation.$reset()
        analysis.$reset()
        learning.$reset()
      }
    },
  )

  // On app startup, validate the persisted session
  session.loadPersistedSession()
}
