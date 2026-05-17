import { app, BrowserWindow, dialog, ipcMain, screen } from 'electron'
import { fileURLToPath } from 'node:url'
import path from 'node:path'
import { pythonManager } from './python-runner'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

process.env.APP_ROOT = path.join(__dirname, '..')

export const VITE_DEV_SERVER_URL = process.env['VITE_DEV_SERVER_URL']
export const MAIN_DIST = path.join(process.env.APP_ROOT, 'dist-electron')
export const RENDERER_DIST = path.join(process.env.APP_ROOT, 'dist')

process.env.VITE_PUBLIC = VITE_DEV_SERVER_URL ? path.join(process.env.APP_ROOT, 'public') : RENDERER_DIST

type WorkspaceTab = 'gallery' | 'classes' | 'clusters' | 'segmentation' | 'analysis'
const TABS: WorkspaceTab[] = ['gallery', 'classes', 'clusters', 'segmentation', 'analysis']
const TAB_TITLES: Record<WorkspaceTab, string> = {
  gallery: 'Gallery',
  classes: 'Classes',
  clusters: 'Clusters',
  segmentation: 'Segmentation',
  analysis: 'Analysis',
}

type WindowRole = 'main' | 'popout'
interface WindowEntry { window: BrowserWindow; role: WindowRole; tabId?: WorkspaceTab }

const windows = new Map<number, WindowEntry>()
const popouts = new Map<WorkspaceTab, BrowserWindow>()
const snapshots = new Map<string, unknown>()
let mainWindow: BrowserWindow | null = null

function registerWindow(win: BrowserWindow, role: WindowRole, tabId?: WorkspaceTab) {
  const wcId = win.webContents.id
  windows.set(wcId, { window: win, role, tabId })
  win.on('closed', () => {
    windows.delete(wcId)
    if (role === 'main') {
      mainWindow = null
    } else if (tabId) {
      popouts.delete(tabId)
      if (mainWindow && !mainWindow.isDestroyed()) {
        mainWindow.webContents.send('popout:closed', tabId)
      }
    }
  })
}

function broadcastSync(senderId: number, storeId: string, state: unknown) {
  snapshots.set(storeId, state)
  for (const entry of windows.values()) {
    if (entry.window.webContents.id === senderId) continue
    if (entry.window.isDestroyed()) continue
    entry.window.webContents.send('sync:apply', storeId, state)
  }
}

function rendererUrl(hash: string) {
  if (VITE_DEV_SERVER_URL) {
    return `${VITE_DEV_SERVER_URL}#${hash}`
  }
  // file:// URL with hash route
  const indexPath = path.join(RENDERER_DIST, 'index.html')
  return `file://${indexPath}#${hash}`
}

function createWindow() {
  const win = new BrowserWindow({
    icon: path.join(process.env.VITE_PUBLIC, 'electron-vite.svg'),
    width: 1440,
    height: 900,
    webPreferences: {
      preload: path.join(__dirname, 'preload.mjs'),
    },
  })

  win.webContents.on('did-finish-load', () => {
    win.webContents.send('main-process-message', (new Date).toLocaleString())
  })

  if (VITE_DEV_SERVER_URL) {
    win.loadURL(VITE_DEV_SERVER_URL)
  } else {
    win.loadFile(path.join(RENDERER_DIST, 'index.html'))
  }

  mainWindow = win
  registerWindow(win, 'main')
}

function createPopoutWindow(tabId: WorkspaceTab, cursorX: number, cursorY: number) {
  if (popouts.has(tabId)) {
    popouts.get(tabId)?.focus()
    return
  }
  // Position the window so the cursor lands just inside the top-left, clamped
  // to the display nearest the cursor.
  const display = screen.getDisplayNearestPoint({ x: cursorX, y: cursorY })
  const width = 1280
  const height = 800
  const offset = 40
  let x = Math.round(cursorX - offset)
  let y = Math.round(cursorY - offset)
  x = Math.max(display.workArea.x, Math.min(x, display.workArea.x + display.workArea.width - width))
  y = Math.max(display.workArea.y, Math.min(y, display.workArea.y + display.workArea.height - height))

  const win = new BrowserWindow({
    icon: path.join(process.env.VITE_PUBLIC, 'electron-vite.svg'),
    width,
    height,
    x,
    y,
    title: `EidoCell — ${TAB_TITLES[tabId]}`,
    webPreferences: {
      preload: path.join(__dirname, 'preload.mjs'),
    },
  })
  win.loadURL(rendererUrl(`/popout/${tabId}`))
  popouts.set(tabId, win)
  registerWindow(win, 'popout', tabId)
  mainWindow?.webContents.send('popout:opened', tabId)
}

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit()
  }
})

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow()
  }
})

ipcMain.handle('select-directory', async () => {
  const result = await dialog.showOpenDialog({ properties: ['openDirectory'] })
  return result.filePaths[0] ?? null
})

ipcMain.handle('select-file', async (_event, options?: { extensions?: string[]; name?: string }) => {
  const filters = options?.extensions?.length
    ? [{ name: options.name ?? 'Files', extensions: options.extensions }]
    : undefined
  const result = await dialog.showOpenDialog({
    properties: ['openFile'],
    filters,
  })
  return result.filePaths[0] ?? null
})

// ── Popout window IPC ───────────────────────────────────────────────────

ipcMain.handle('popout:open', (_event, tabId: WorkspaceTab, x: number, y: number) => {
  if (!TABS.includes(tabId)) return { ok: false, reason: 'invalid-tab' }
  if (popouts.has(tabId)) {
    popouts.get(tabId)?.focus()
    return { ok: false, reason: 'already-open' }
  }
  // Enforce: at least one tab must remain in the main window.
  if (popouts.size + 1 >= TABS.length) {
    return { ok: false, reason: 'min-one-tab' }
  }
  createPopoutWindow(tabId, x, y)
  return { ok: true }
})

ipcMain.handle('popout:focus', (_event, tabId: WorkspaceTab) => {
  const win = popouts.get(tabId)
  if (!win) return false
  if (win.isMinimized()) win.restore()
  win.focus()
  return true
})

ipcMain.handle('popout:list', () => [...popouts.keys()])

// ── Pinia state sync relay ──────────────────────────────────────────────

ipcMain.on('sync:broadcast', (event, storeId: string, state: unknown) => {
  broadcastSync(event.sender.id, storeId, state)
})

ipcMain.handle('sync:snapshot', (_event, storeId: string) => {
  return snapshots.get(storeId) ?? null
})

// Cross-window data invalidation bus. The emitter already fetched on its
// side (its action is responsible) — we just notify the other windows.
ipcMain.on('data:emit', (event, topics: string[]) => {
  for (const entry of windows.values()) {
    if (entry.window.webContents.id === event.sender.id) continue
    if (entry.window.isDestroyed()) continue
    entry.window.webContents.send('data:apply', topics)
  }
})

app.whenReady().then(() => {
  pythonManager.start()
  createWindow()
})

app.on('will-quit', () => {
  pythonManager.stop()
})
