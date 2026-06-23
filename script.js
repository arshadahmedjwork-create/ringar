import {
  HandLandmarker,
  PoseLandmarker,
  FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/vision_bundle.mjs";

const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const enableWebcamButton = document.getElementById("webcamButton");
const loadingText = document.getElementById("loading");

const ringImg = document.getElementById("ring-image");
const necklessImg = document.getElementById("neckless-image");

const modeRingBtn = document.getElementById("mode-ring");
const modeNecklessBtn = document.getElementById("mode-neckless");

let handLandmarker = undefined;
let poseLandmarker = undefined;
let webcamRunning = false;
let lastVideoTime = -1;
let currentMode = "ring"; // "ring" or "neckless"
let handResults = undefined;
let poseResults = undefined;

// Wait for both landmarkers to finish loading
const createLandmarkers = async () => {
  try {
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
    );

    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        delegate: "CPU"
      },
      runningMode: "VIDEO",
      numHands: 1
    });

    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
        delegate: "CPU"
      },
      runningMode: "VIDEO"
    });

    loadingText.style.display = "none";
    enableWebcamButton.disabled = false;
    
    // Automatically prompt the user to allow the camera
    if (hasGetUserMedia()) {
      enableCam();
    }
  } catch (error) {
    console.error("Error loading MediaPipe:", error);
    loadingText.innerText = "Error: " + error.message;
    loadingText.style.color = "red";
  }
};

createLandmarkers();

// Mode switching
modeRingBtn.addEventListener("click", () => {
  currentMode = "ring";
  modeRingBtn.classList.add("active");
  modeNecklessBtn.classList.remove("active");
});

modeNecklessBtn.addEventListener("click", () => {
  currentMode = "neckless";
  modeNecklessBtn.classList.add("active");
  modeRingBtn.classList.remove("active");
});

// Check if webcam access is supported
const hasGetUserMedia = () => !!navigator.mediaDevices?.getUserMedia;

if (hasGetUserMedia()) {
  enableWebcamButton.addEventListener("click", enableCam);
} else {
  console.warn("getUserMedia() is not supported by your browser");
}

function enableCam(event) {
  if (!handLandmarker || !poseLandmarker) {
    console.log("Wait! models not loaded yet.");
    return;
  }

  if (webcamRunning === true) {
    webcamRunning = false;
    enableWebcamButton.innerText = "ENABLE PREDICTIONS";
  } else {
    webcamRunning = true;
    enableWebcamButton.innerText = "DISABLE PREDICTIONS";
  }

  const constraints = {
    video: { width: 640, height: 480 }
  };

  navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
    video.srcObject = stream;
    video.addEventListener("loadeddata", predictWebcam);
    video.classList.add("active");
  });
}

async function predictWebcam() {
  canvasElement.style.width = video.videoWidth + "px";
  canvasElement.style.height = video.videoHeight + "px";
  canvasElement.width = video.videoWidth;
  canvasElement.height = video.videoHeight;

  if (webcamRunning === true) {
    let startTimeMs = performance.now();

    if (lastVideoTime !== video.currentTime) {
      if (currentMode === "ring") {
        handResults = handLandmarker.detectForVideo(video, startTimeMs);
      } else if (currentMode === "neckless") {
        poseResults = poseLandmarker.detectForVideo(video, startTimeMs);
      }
      lastVideoTime = video.currentTime;
    }

    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    if (currentMode === "ring" && handResults && handResults.landmarks) {
      for (const landmarks of handResults.landmarks) {
        const p13 = landmarks[13]; // MCP (base of finger)
        const p14 = landmarks[14]; // PIP (middle joint)

        if (p13 && p14) {
          const targetCx = ((p13.x + p14.x) / 2) * canvasElement.width;
          const targetCy = ((p13.y + p14.y) / 2) * canvasElement.height;

          const dx = (p14.x - p13.x) * canvasElement.width;
          const dy = (p14.y - p13.y) * canvasElement.height;
          const distance = Math.sqrt(dx * dx + dy * dy);

          const angle = Math.atan2(dy, dx);
          const targetWidth = distance * 0.8;
          const targetHeight = targetWidth * (ringImg.naturalHeight / ringImg.naturalWidth);

          canvasCtx.save();
          canvasCtx.translate(targetCx, targetCy);
          canvasCtx.rotate(angle + Math.PI / 2);
          canvasCtx.drawImage(ringImg, -targetWidth / 2, -targetHeight / 2, targetWidth, targetHeight);
          canvasCtx.restore();
        }
      }
    } else if (currentMode === "neckless" && poseResults && poseResults.landmarks) {
      for (const landmarks of poseResults.landmarks) {
        const nose = landmarks[0];
        const leftShoulder = landmarks[11];
        const rightShoulder = landmarks[12];

        if (leftShoulder && rightShoulder && nose) {
          // Calculate shoulder midpoint
          const shoulderMidX = (leftShoulder.x + rightShoulder.x) / 2;
          const dx = (rightShoulder.x - leftShoulder.x) * canvasElement.width;
          const dy = (rightShoulder.y - leftShoulder.y) * canvasElement.height;
          let shoulderDist = Math.sqrt(dx * dx + dy * dy);

          if (shoulderDist < canvasElement.width * 0.1) {
            shoulderDist = canvasElement.width * 0.4;
          }

          // Since shoulders can get cut off by the webcam frame causing bad coordinates, 
          // we'll calculate the vertical position relative to the nose instead.
          // Drop it below the nose by an amount proportional to the shoulder width.
          let targetCx = (nose.x * 0.5 + shoulderMidX * 0.5) * canvasElement.width;
          let targetCy = (nose.y * canvasElement.height) + (shoulderDist * 0.55);

          let targetWidth = shoulderDist * 0.7; // Slightly scale down to fit the neck better
          let targetHeight = targetWidth;
          if (necklessImg.naturalWidth > 0) {
            targetHeight = targetWidth * (necklessImg.naturalHeight / necklessImg.naturalWidth);
          }

          // Exponential Moving Average (EMA) for Smoothing
          const alpha = 0.15; // Lower is smoother but lags slightly
          if (typeof window.smoothNeckX === 'undefined' || Math.abs(window.smoothNeckX - targetCx) > 100) {
            window.smoothNeckX = targetCx;
            window.smoothNeckY = targetCy;
            window.smoothNeckW = targetWidth;
          } else {
            window.smoothNeckX = window.smoothNeckX + alpha * (targetCx - window.smoothNeckX);
            window.smoothNeckY = window.smoothNeckY + alpha * (targetCy - window.smoothNeckY);
            window.smoothNeckW = window.smoothNeckW + alpha * (targetWidth - window.smoothNeckW);
          }

          targetHeight = window.smoothNeckW * (necklessImg.naturalHeight / necklessImg.naturalWidth);

          canvasCtx.save();
          canvasCtx.translate(window.smoothNeckX, window.smoothNeckY);
          // Shift Y up by half height so the top of the necklace is at the tracking point
          canvasCtx.drawImage(necklessImg, -window.smoothNeckW / 2, -targetHeight / 2, window.smoothNeckW, targetHeight);
          canvasCtx.restore();
        }
      }
    }

    window.requestAnimationFrame(predictWebcam);
  }
}
