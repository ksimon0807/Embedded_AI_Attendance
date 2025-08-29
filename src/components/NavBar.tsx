import React from 'react'

export default function NavBar({ route, setRoute }: { route: 'student' | 'admin'; setRoute: (r: 'student' | 'admin') => void }) {
  return (
    <header className="bg-white shadow">
      <div className="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between">
        <div className="text-lg font-semibold">AI Attendance</div>
        <nav className="space-x-2">
          <button
            onClick={() => setRoute('student')}
            className={`px-3 py-1 rounded ${route === 'student' ? 'bg-green-600 text-white' : 'text-gray-700 hover:bg-gray-100'}`}
          >
            Student
          </button>
          <button
            onClick={() => setRoute('admin')}
            className={`px-3 py-1 rounded ${route === 'admin' ? 'bg-green-600 text-white' : 'text-gray-700 hover:bg-gray-100'}`}
          >
            Admin
          </button>
        </nav>
      </div>
    </header>
  )
}
