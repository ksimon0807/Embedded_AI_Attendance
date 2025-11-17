import React, { useState, useEffect, useCallback } from "react";

// --- Configuration ---
// IMPORTANT: Make sure this URL matches where your Python backend is running.
const BACKEND_URL = "https://127.0.0.1:5000";
const HEALTH_CHECK_INTERVAL = 5000; // Check backend status every 5 seconds.

// --- Reusable UI Components ---

/**
 * A toast notification component for displaying success or error messages.
 */
const Toast = ({ toast, onClose }) => (
  <div className="max-w-sm w-full bg-white shadow-lg rounded-lg pointer-events-auto ring-1 ring-black ring-opacity-5 overflow-hidden animate-fade-in-right">
    <div className="p-4">
      <div className="flex items-start">
        <div className="flex-shrink-0">
          {toast.type === "success" && (
            <svg
              className="h-6 w-6 text-green-400"
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
          )}
          {toast.type === "error" && (
            <svg
              className="h-6 w-6 text-red-400"
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
          )}
        </div>
        <div className="ml-3 w-0 flex-1 pt-0.5">
          <p className="text-sm font-medium text-gray-900">{toast.title}</p>
          {toast.description && (
            <p className="mt-1 text-sm text-gray-500">{toast.description}</p>
          )}
        </div>
        <div className="ml-4 flex-shrink-0 flex">
          <button
            onClick={onClose}
            className="bg-white rounded-md inline-flex text-gray-400 hover:text-gray-500 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
          >
            <span className="sr-only">Close</span>
            <svg
              className="h-5 w-5"
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 20 20"
              fill="currentColor"
            >
              <path
                fillRule="evenodd"
                d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
                clipRule="evenodd"
              />
            </svg>
          </button>
        </div>
      </div>
    </div>
  </div>
);

/**
 * The main navigation bar with view switching and a backend status indicator.
 */
const NavBar = ({ route, setRoute, backendStatus }) => {
  const StatusIndicator = () => (
    <div className="flex items-center space-x-2">
      <span
        className={`h-3 w-3 rounded-full animate-pulse ${
          backendStatus === "online" ? "bg-green-400" : "bg-red-500"
        }`}
      ></span>
      <span className="text-sm font-medium text-gray-300">
        Backend:{" "}
        <span
          className={
            backendStatus === "online" ? "text-green-300" : "text-red-300"
          }
        >
          {backendStatus}
        </span>
      </span>
    </div>
  );

  return (
    <nav className="bg-gray-800 shadow-md sticky top-0 z-40">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center">
            <span className="text-white text-xl font-bold">
              Smart Attendance
            </span>
          </div>
          <div className="hidden md:block">
            <div className="ml-10 flex items-baseline space-x-4">
              <button
                onClick={() => setRoute("student")}
                className={`${
                  route === "student"
                    ? "bg-gray-900 text-white"
                    : "text-gray-300 hover:bg-gray-700"
                } px-3 py-2 rounded-md text-sm font-medium`}
              >
                Student View
              </button>
              <button
                onClick={() => setRoute("admin")}
                className={`${
                  route === "admin"
                    ? "bg-gray-900 text-white"
                    : "text-gray-300 hover:bg-gray-700"
                } px-3 py-2 rounded-md text-sm font-medium`}
              >
                Admin Dashboard
              </button>
            </div>
          </div>
          <div className="hidden md:block">
            <StatusIndicator />
          </div>
        </div>
      </div>
    </nav>
  );
};

// --- Page Components ---

/**
 * The dashboard for the student view, showing the camera feed and attendance button.
 */
const StudentDashboard = ({ showToast, backendStatus }) => {
  const [isMarking, setIsMarking] = useState(false);
  const [faceDetected, setFaceDetected] = useState(false);
  const [isDetecting, setIsDetecting] = useState(false);
  const [detectionCount, setDetectionCount] = useState(0);
  const [videoFeedError, setVideoFeedError] = useState(false);
  const videoFeedUrl = `${BACKEND_URL}/video_feed`;

  // Function to check if a face is detected
  const checkFaceDetection = useCallback(async () => {
    if (backendStatus !== "online" || isDetecting) return;

    setIsDetecting(true);
    try {
      const controller = new AbortController();
      const t = setTimeout(() => controller.abort(), 2500);
      const response = await fetch(`${BACKEND_URL}/detect_face`, {
        signal: controller.signal,
      });
      clearTimeout(t);
      const data = await response.json();

      if (data.status === "success") {
        if (data.face_detected) {
          // Increment detection count when face is found
          setDetectionCount((prev) => {
            const newCount = prev + 1;
            // Require 2 consecutive detections to confirm face is stable
            if (newCount >= 2) {
              setFaceDetected(true);
            }
            return newCount;
          });
        } else {
          // Reset detection count when no face is found
          setDetectionCount(0);
          setFaceDetected(false);
        }
      } else {
        setDetectionCount(0);
        setFaceDetected(false);
      }
    } catch (error) {
      console.error("Error checking face detection:", error);
      setDetectionCount(0);
      setFaceDetected(false);
    } finally {
      // Add a delay to show the checking state longer for better UX
      setTimeout(() => {
        setIsDetecting(false);
      }, 1500); // Show "checking" for 1.5 seconds for better visibility
    }
  }, [backendStatus]);

  // Check face detection every 3 seconds when backend is online (increased interval for stability)
  useEffect(() => {
    if (backendStatus !== "online") {
      setFaceDetected(false);
      setDetectionCount(0);
      return;
    }

    // Reset video feed error when backend comes online
    setVideoFeedError(false);

    const intervalId = setInterval(checkFaceDetection, 3000);
    return () => clearInterval(intervalId);
  }, [backendStatus, checkFaceDetection]);

  const handleMarkAttendance = async () => {
    setIsMarking(true);
    try {
      // This request triggers the face recognition on the backend.
      const response = await fetch(`${BACKEND_URL}/mark_attendance`, {
        method: "POST",
      });
      const data = await response.json();

      if (data.status === "success") {
        showToast({
          title: data.message || "Attendance Marked!",
          type: "success",
        });
      } else {
        showToast({
          title: "Attendance Failed",
          description: data.message, // e.g., "No face detected" or "Unknown person"
          type: "error",
        });
      }
    } catch (error) {
      console.error("Error marking attendance:", error);
      showToast({
        title: "Connection Error",
        description: "Could not connect to the backend server.",
        type: "error",
      });
    }
    setIsMarking(false);
  };

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold text-gray-800">Student Dashboard</h1>
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 bg-white p-4 rounded-lg shadow-md border">
          <h2 className="text-xl font-semibold mb-2 text-gray-700">
            Live Camera Feed
          </h2>
          <div className="w-full aspect-video bg-gray-900 rounded-md flex items-center justify-center border overflow-hidden relative">
            {backendStatus === "online" ? (
              <>
                {videoFeedError ? (
                  <div className="text-gray-400 text-center p-4">
                    <h3 className="text-lg font-semibold">Camera Feed Error</h3>
                    <p className="text-sm mt-2">
                      Unable to load video feed. Please check ESP32 camera
                      connection.
                    </p>
                    <button
                      onClick={() => {
                        setVideoFeedError(false);
                        // Force image reload by adding timestamp
                        const img = document.querySelector(
                          'img[alt="Live ESP32-CAM Feed"]'
                        );
                        if (img) {
                          img.src = `${videoFeedUrl}?t=${Date.now()}`;
                        }
                      }}
                      className="mt-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
                    >
                      Retry
                    </button>
                  </div>
                ) : (
                  <img
                    src={videoFeedUrl}
                    alt="Live ESP32-CAM Feed"
                    className="w-full h-full object-cover"
                    onError={() => {
                      console.error("Video feed failed to load");
                      setVideoFeedError(true);
                    }}
                    onLoad={() => {
                      setVideoFeedError(false);
                    }}
                  />
                )}
                {/* Face Detection Overlay */}
                {!videoFeedError && isDetecting && (
                  <div className="absolute top-4 right-4 bg-blue-500 text-white px-3 py-1 rounded-full text-sm font-medium shadow-lg animate-pulse">
                    🔍 Analyzing...
                  </div>
                )}
                {!videoFeedError && !isDetecting && faceDetected && (
                  <div className="absolute top-4 right-4 bg-green-500 text-white px-3 py-1 rounded-full text-sm font-medium shadow-lg">
                    ✓ Face Detected
                  </div>
                )}
                {!videoFeedError &&
                  !isDetecting &&
                  !faceDetected &&
                  backendStatus === "online" && (
                    <div className="absolute top-4 right-4 bg-red-500 text-white px-3 py-1 rounded-full text-sm font-medium shadow-lg">
                      ✗ No Face
                    </div>
                  )}
              </>
            ) : (
              <div className="text-gray-400 text-center p-4">
                <h3 className="text-lg font-semibold">Camera Offline</h3>
                <p>
                  The backend server is not running, so the camera feed cannot
                  be displayed.
                </p>
              </div>
            )}
          </div>
        </div>
        <div className="bg-white p-6 rounded-lg shadow-md border">
          <h2 className="text-xl font-semibold mb-4 text-gray-700">
            Mark Your Attendance
          </h2>

          {/* Face Detection Status */}
          <div className="mb-4 p-3 rounded-lg border">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-gray-700">
                Face Detection Status:
              </span>
              <div className="flex items-center space-x-2">
                <div
                  className={`h-3 w-3 rounded-full ${
                    faceDetected ? "bg-green-500" : "bg-red-500"
                  }`}
                ></div>
                <span
                  className={`text-sm font-medium ${
                    faceDetected ? "text-green-600" : "text-red-600"
                  }`}
                >
                  {faceDetected ? "Face Detected" : "No Face Detected"}
                </span>
              </div>
            </div>

            {/* Detection Progress Bar */}
            {isDetecting && (
              <div className="mt-2">
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-blue-600 h-2 rounded-full animate-pulse"
                    style={{ width: "100%" }}
                  ></div>
                </div>
                <p className="text-xs text-blue-600 mt-1 text-center">
                  Analyzing camera frame...
                </p>
              </div>
            )}

            {/* Detection Stability Indicator */}
            {!isDetecting && backendStatus === "online" && (
              <div className="mt-2">
                <div className="flex items-center justify-between text-xs text-gray-500">
                  <span>Detection Stability:</span>
                  <span className="font-medium">
                    {detectionCount >= 2
                      ? "Stable ✓"
                      : `${detectionCount}/2 detections`}
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2 mt-1">
                  <div
                    className={`h-2 rounded-full transition-all duration-300 ${
                      detectionCount >= 2 ? "bg-green-500" : "bg-yellow-400"
                    }`}
                    style={{ width: `${Math.min(detectionCount * 50, 100)}%` }}
                  ></div>
                </div>
              </div>
            )}

            <div className="flex items-center justify-between mt-2">
              <div className="text-xs text-gray-500">
                Auto-check every 3 seconds
              </div>
              <button
                onClick={checkFaceDetection}
                disabled={isDetecting || backendStatus !== "online"}
                className="text-xs px-3 py-1 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
              >
                {isDetecting ? "Analyzing..." : "Check Now"}
              </button>
            </div>
          </div>

          <p className="text-sm text-gray-600 mb-4">
            Position your face clearly in the camera view. The button will be
            enabled once a face is detected and confirmed stable (2 consecutive
            detections).
          </p>

          <button
            onClick={handleMarkAttendance}
            disabled={isMarking || backendStatus !== "online" || !faceDetected}
            className="w-full px-4 py-3 bg-indigo-600 text-white font-bold rounded-lg shadow-md hover:bg-indigo-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-all duration-300 ease-in-out focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-opacity-50"
          >
            {isMarking ? "Processing..." : "Mark My Attendance"}
          </button>

          {backendStatus !== "online" && (
            <p className="text-xs text-red-600 mt-2 text-center">
              Button is disabled because the backend is offline.
            </p>
          )}

          {backendStatus === "online" && !faceDetected && (
            <p className="text-xs text-orange-600 mt-2 text-center">
              Button is disabled because no face is detected in the camera view.
            </p>
          )}
        </div>
      </div>
    </div>
  );
};

/**
 * The dashboard for the admin view, showing a table of all attendance records.
 */
const AdminDashboard = ({ backendStatus }) => {
  const [records, setRecords] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchAttendance = useCallback(async () => {
    if (backendStatus !== "online") {
      setRecords([]);
      setError("Backend is offline. Cannot fetch attendance data.");
      setLoading(false);
      return;
    }
    setLoading(true);
    try {
      const response = await fetch(`${BACKEND_URL}/get_attendance`);
      if (!response.ok) throw new Error("Network response was not ok");
      const data = await response.json();
      setRecords(data);
      setError(null);
    } catch (e) {
      console.error("Failed to fetch attendance:", e);
      setError(
        "Failed to fetch attendance data. Is the backend server running?"
      );
    } finally {
      setLoading(false);
    }
  }, [backendStatus]);

  useEffect(() => {
    fetchAttendance();
  }, [fetchAttendance]);

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold text-gray-800">Admin Dashboard</h1>
        <button
          onClick={fetchAttendance}
          disabled={loading || backendStatus !== "online"}
          className="px-4 py-2 bg-gray-700 text-white text-sm font-medium rounded-md hover:bg-gray-800 disabled:bg-gray-400"
        >
          {loading ? "Refreshing..." : "Refresh Data"}
        </button>
      </div>
      <div className="bg-white p-4 rounded-lg shadow-md border overflow-hidden">
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th
                  scope="col"
                  className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                >
                  Name
                </th>
                <th
                  scope="col"
                  className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                >
                  Roll
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {loading && (
                <tr>
                  <td colSpan="3" className="text-center p-4 text-gray-500">
                    Loading...
                  </td>
                </tr>
              )}
              {error && (
                <tr>
                  <td
                    colSpan="3"
                    className="text-center p-4 text-red-600 font-semibold"
                  >
                    {error}
                  </td>
                </tr>
              )}
              {!loading && !error && records.length === 0 && (
                <tr>
                  <td colSpan="3" className="text-center p-4 text-gray-500">
                    No attendance records found.
                  </td>
                </tr>
              )}
              {!loading &&
                !error &&
                records.map((r, index) => (
                  <tr key={index} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                      {r.Name}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {r.Roll}
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

// --- The Main App Component That Ties Everything Together ---

export default function App() {
  const [route, setRoute] = useState("student");
  const [toasts, setToasts] = useState([]);
  const [backendStatus, setBackendStatus] = useState("offline");

  // Periodically check the health of the backend server.
  useEffect(() => {
    const checkBackendStatus = async () => {
      try {
        const response = await fetch(`${BACKEND_URL}/health`);
        setBackendStatus(response.ok ? "online" : "offline");
      } catch (error) {
        setBackendStatus("offline");
      }
    };

    checkBackendStatus(); // Initial check
    const intervalId = setInterval(checkBackendStatus, HEALTH_CHECK_INTERVAL);

    return () => clearInterval(intervalId); // Cleanup on component unmount
  }, []);

  // Function to show a toast notification.
  const showToast = useCallback((toastData) => {
    const id = Date.now();
    setToasts((currentToasts) => [{ id, ...toastData }, ...currentToasts]);
    setTimeout(() => {
      setToasts((currentToasts) => currentToasts.filter((t) => t.id !== id));
    }, 5000);
  }, []);

  const removeToast = (id) => {
    setToasts((currentToasts) => currentToasts.filter((t) => t.id !== id));
  };

  return (
    <div className="min-h-screen bg-gray-100 font-sans">
      <NavBar route={route} setRoute={setRoute} backendStatus={backendStatus} />

      <main className="max-w-7xl mx-auto p-4 sm:p-6 lg:p-8">
        {route === "student" ? (
          <StudentDashboard
            showToast={showToast}
            backendStatus={backendStatus}
          />
        ) : (
          <AdminDashboard backendStatus={backendStatus} />
        )}
      </main>

      {/* Global container for all toast notifications */}
      <div
        aria-live="assertive"
        className="fixed inset-0 flex items-end px-4 py-6 pointer-events-none sm:p-6 sm:items-start z-50"
      >
        <div className="w-full flex flex-col items-center space-y-4 sm:items-end">
          {toasts.map((t) => (
            <Toast key={t.id} toast={t} onClose={() => removeToast(t.id)} />
          ))}
        </div>
      </div>
    </div>
  );
}
