import { spawn, ChildProcess } from 'node:child_process'
import path from 'node:path'
import { app } from 'electron'

const PY_HOST = '127.0.0.1'
const PY_PORT = 8000
const isDev = !app.isPackaged

class PythonManager {
  private process: ChildProcess | null = null
  private isStopping = false

  private getConfig() {
    if (isDev) {
      return {
        command: 'poetry',
        args: ['run', 'uvicorn', 'main:app', '--host', PY_HOST, '--port', `${PY_PORT}`, '--reload'],
        cwd: path.join(process.env.APP_ROOT!, '..', 'eidocell-backend'),
        shell: true,
      }
    } else {
      const exeName = process.platform === 'win32' ? 'eidocell-backend.exe' : 'eidocell-backend'
      return {
        command: path.join(process.resourcesPath, 'eidocell-backend', exeName),
        args: [],
        cwd: path.join(process.resourcesPath, 'eidocell-backend'),
        shell: false,
      }
    }
  }

  start() {
    const { command, args, cwd, shell } = this.getConfig()
    console.log(`[PythonManager] Spawning: ${command} ${args.join(' ')} (cwd: ${cwd})`)

    this.process = spawn(command, args, { cwd, shell, stdio: 'pipe' })

    this.process.stdout?.on('data', (data: Buffer) => {
      console.log(`[PY]: ${data.toString().trim()}`)
    })

    this.process.stderr?.on('data', (data: Buffer) => {
      console.error(`[PY ERR]: ${data.toString().trim()}`)
    })

    this.process.on('close', (code) => {
      console.log(`[PythonManager] Process exited with code ${code}`)
      this.process = null

      if (!this.isStopping && !isDev && code !== 0) {
        console.log('[PythonManager] Process crashed. Restarting in 3s...')
        setTimeout(() => this.start(), 3000)
      }
    })
  }

  stop() {
    this.isStopping = true
    if (this.process?.pid) {
      console.log('[PythonManager] Killing Python process tree...')
      // Use tree-kill to kill the entire process tree (poetry -> uvicorn -> workers)
      import('tree-kill').then(({ default: treeKill }) => {
        treeKill(this.process!.pid!, 'SIGTERM', (err) => {
          if (err) {
            console.error('[PythonManager] Failed to kill process tree:', err)
          } else {
            console.log('[PythonManager] Process tree killed.')
          }
        })
      }).catch(() => {
        // Fallback if tree-kill not available
        this.process?.kill('SIGTERM')
      })
      this.process = null
    }
  }
}

export const pythonManager = new PythonManager()
