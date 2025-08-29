import React from 'react'
import VideoPlaceholder from '../components/VideoPlaceholder'
import { Toast } from '../App'

type Props = {
  showToast: (t: Omit<Toast, 'id'>) => void
}

export default function StudentDashboard({ showToast }: Props) {
  const handleSimulate = () => {
    showToast({
      title: 'Attendance marked successfully',
      description: 'Name: Aisha Sharma | Roll: 21 | Subject: Computer Science',
    })
  }

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold">Student Dashboard</h1>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <VideoPlaceholder />

        <div className="bg-white p-4 rounded shadow">
          <h2 className="font-medium mb-2">Actions</h2>
          <p className="text-sm text-gray-600 mb-4">Use the button below to simulate a face detection event and mark attendance.</p>
          <button
            onClick={handleSimulate}
            className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
          >
            Simulate Face Detection
          </button>
        </div>
      </div>
    </div>
  )
}
