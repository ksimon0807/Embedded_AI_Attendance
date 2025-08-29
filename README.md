# Attendance UI (Vite + React + Tailwind)

Beginner-friendly frontend scaffold for an AI-powered automatic attendance system.

What you get
- Student dashboard with a video placeholder and "Simulate Face Detection" button that shows a toast.
- Admin dashboard with a mock attendance table and filters.
- Simple top nav and Tailwind styling.

Quick start (Windows PowerShell)

1. Open PowerShell in the `attendance-ui` folder

```powershell
npm install
npm run dev
```

2. Open the printed local URL (usually http://localhost:5173)

Notes
- This scaffold uses mock data and placeholders. Backend hooks (ESP32 stream, Supabase fetch, attendance API) can be added later in the components and pages.
- Tailwind is configured in `tailwind.config.cjs` and styles are in `src/styles/index.css`.

Files of interest:
- `src/pages/StudentDashboard.tsx`
- `src/pages/AdminDashboard.tsx`
- `src/components/VideoPlaceholder.tsx`
- `src/data/mockData.ts`

Next steps you might want:
- Wire the VideoPlaceholder to an actual video stream URL (iframe or <video> tag)
- Replace mock data with a Supabase fetch in `AdminDashboard`
- Add authentication for admin routes
