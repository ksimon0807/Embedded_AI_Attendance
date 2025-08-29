export type StudentRecord = {
  name: string
  roll: string
  subject: string
  attendance: number
  lastMarked: string
}

export const mockStudents: StudentRecord[] = [
  { name: 'Aisha Sharma', roll: '21', subject: 'Computer Science', attendance: 96, lastMarked: '2025-08-26' },
  { name: 'Rahul Verma', roll: '12', subject: 'Mathematics', attendance: 88, lastMarked: '2025-08-25' },
  { name: 'Priya Singh', roll: '05', subject: 'Physics', attendance: 92, lastMarked: '2025-08-24' },
  { name: 'Aisha Sharma', roll: '21', subject: 'Mathematics', attendance: 94, lastMarked: '2025-08-20' },
  { name: 'Rahul Verma', roll: '12', subject: 'Computer Science', attendance: 85, lastMarked: '2025-08-22' },
]
