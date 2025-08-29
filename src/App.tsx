import React, { useState } from 'react'
import NavBar from './components/NavBar'
import StudentDashboard from './pages/StudentDashboard'
import AdminDashboard from './pages/AdminDashboard'
import { mockStudents } from './data/mockData'

export type Toast = {
  id: number
  title: string
  description?: string
}

export default function App() {
  const [route, setRoute] = useState<'student' | 'admin'>('student')
  const [toasts, setToasts] = useState<Toast[]>([])

  const showToast = (t: Omit<Toast, 'id'>) => {
    const id = Date.now()
    setToasts((s) => [{ id, ...t }, ...s])
    setTimeout(() => {
      setToasts((s) => s.filter((x) => x.id !== id))
    }, 4000)
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <NavBar route={route} setRoute={setRoute} />

      <main className="p-6">
        {route === 'student' ? (
          <StudentDashboard showToast={showToast} />
        ) : (
          <AdminDashboard data={mockStudents} />
        )}
      </main>

      {/* Toasts */}
      <div className="fixed top-4 right-4 space-y-2 z-50">
        {toasts.map((t) => (
          <div
            key={t.id}
            className="max-w-sm bg-white p-3 rounded shadow border-l-4 border-green-500"
          >
            <div className="font-semibold">{t.title}</div>
            {t.description && <div className="text-sm text-gray-600">{t.description}</div>}
          </div>
        ))}
      </div>
    </div>
  )
}
