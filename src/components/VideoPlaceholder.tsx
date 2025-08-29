import React from 'react'

export default function VideoPlaceholder() {
  return (
    <div className="bg-white p-4 rounded shadow">
      <h2 className="font-medium mb-2">Live Camera Feed</h2>
      <div className="w-full h-64 bg-gray-100 rounded flex items-center justify-center border">
        <div className="text-gray-500 text-center">
          <div className="mb-2">ESP32-CAM stream will appear here</div>
          <div className="text-xs">(Placeholder)</div>
        </div>
      </div>
      <p className="text-xs text-gray-500 mt-2">Tip: plug your ESP32-CAM stream URL into the VideoPlaceholder component later.</p>
    </div>
  )
}
