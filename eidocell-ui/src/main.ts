import { createApp } from 'vue'
import { createPinia } from 'pinia'
import './style.css'
import App from './App.vue'
import router from './router'
import { windowSyncPlugin } from './plugins/windowSync'
import { initDataBus } from './lib/dataBus'
import { usePopoutsStore, type WorkspaceTab } from './stores/popouts'

const app = createApp(App)
const pinia = createPinia()
pinia.use(windowSyncPlugin)
app.use(pinia)
app.use(router)

const isPopout = window.location.hash.startsWith('#/popout/')

app.mount('#app').$nextTick(() => {
  if (!window.ipcRenderer) return

  initDataBus()

  window.ipcRenderer.on('main-process-message', (_event, message) => {
    console.log(message)
  })

  // Only the main window owns popout lifecycle state. Popout windows
  // receive their `popouts` store updates via the sync plugin.
  if (isPopout) return

  const popouts = usePopoutsStore()

  window.ipcRenderer.invoke('popout:list').then((tabs) => {
    if (Array.isArray(tabs)) {
      popouts.set(tabs as WorkspaceTab[])
    }
  })

  window.ipcRenderer.on('popout:opened', (_event, tabId: WorkspaceTab) => {
    popouts.add(tabId)
  })

  window.ipcRenderer.on('popout:closed', (_event, tabId: WorkspaceTab) => {
    popouts.remove(tabId)
  })
})
