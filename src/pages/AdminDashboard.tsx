import React, { useMemo, useState } from 'react'
import { StudentRecord } from '../data/mockData'

type Props = { data: StudentRecord[] }

export default function AdminDashboard({ data }: Props) {
  const subjects = useMemo(() => Array.from(new Set(data.map((d) => d.subject))), [data])
  const students = useMemo(() => Array.from(new Set(data.map((d) => d.name))), [data])

  const [subjectFilter, setSubjectFilter] = useState<string>('All')
  const [studentFilter, setStudentFilter] = useState<string>('All')

  const filtered = data.filter((d) => {
    if (subjectFilter !== 'All' && d.subject !== subjectFilter) return false
    if (studentFilter !== 'All' && d.name !== studentFilter) return false
    return true
  })

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold">Admin Dashboard</h1>

      <div className="flex flex-col md:flex-row gap-4">
        <div className="bg-white p-4 rounded shadow flex items-center gap-4">
          <div>
            <label className="text-sm block">Subject</label>
            <select
              className="mt-1 border rounded px-2 py-1"
              value={subjectFilter}
              onChange={(e) => setSubjectFilter(e.target.value)}
            >
              <option>All</option>
              {subjects.map((s) => (
                <option key={s}>{s}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="text-sm block">Student</label>
            <select
              className="mt-1 border rounded px-2 py-1"
              value={studentFilter}
              onChange={(e) => setStudentFilter(e.target.value)}
            >
              <option>All</option>
              {students.map((s) => (
                <option key={s}>{s}</option>
              ))}
            </select>
          </div>
        </div>
      </div>

      <div className="bg-white p-4 rounded shadow overflow-auto">
        <table className="min-w-full table-auto">
          <thead>
            <tr className="text-left text-sm text-gray-600">
              <th className="px-3 py-2">Student Name</th>
              <th className="px-3 py-2">Roll No.</th>
              <th className="px-3 py-2">Subject</th>
              <th className="px-3 py-2">Attendance %</th>
              <th className="px-3 py-2">Last Marked Date</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((r) => (
              <tr key={`${r.roll}-${r.subject}`} className="border-t">
                <td className="px-3 py-2">{r.name}</td>
                <td className="px-3 py-2">{r.roll}</td>
                <td className="px-3 py-2">{r.subject}</td>
                <td className="px-3 py-2">{r.attendance}%</td>
                <td className="px-3 py-2">{r.lastMarked}</td>
              </tr>
            ))}
          </tbody>
        </table>

        {filtered.length === 0 && <div className="p-4 text-gray-600">No records match the filters.</div>}
      </div>
    </div>
  )
}
