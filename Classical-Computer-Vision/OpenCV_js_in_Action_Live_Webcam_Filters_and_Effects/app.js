console.log("Waiting for OpenCV...");

cv['onRuntimeInitialized'] = async () => {
    console.log("OpenCV ready!");

    // DOM
    const video = document.getElementById("video");
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");

    const fpsText = document.getElementById('fps');
    const memoryText = document.getElementById('memory');
    const filterSelect = document.getElementById('filter');
    const intensitySlider = document.getElementById('intensity');
    const intensityValue = document.getElementById('intensity-value');

    let faceCache = [];     // store last detected faces
    let detectCounter = 0;  // run detector every N frames
    const DETECT_INTERVAL = 4; // detect every 4th frame

    const ellipseScale = 1.25;   // increase mask radius



    intensitySlider.addEventListener("input", () => {
        intensityValue.textContent = intensitySlider.value;
    });

    // Canvas size
    const W = 640;
    const H = 480;
    canvas.width = W;
    canvas.height = H;

    // Allocate reusable Mats
    const src = new cv.Mat(H, W, cv.CV_8UC4);
    const dst = new cv.Mat(H, W, cv.CV_8UC4);
    const gray = new cv.Mat(H, W, cv.CV_8UC1);
    const color = new cv.Mat(H, W, cv.CV_8UC3);
    const edges = new cv.Mat(H, W, cv.CV_8UC1);
    const bgr = new cv.Mat(H, W, cv.CV_8UC3);

    // Haarcascade
    let faceCascade = null;

    async function loadHaarCascade() {
        console.log("Loading Haarcascade...");
        faceCascade = new cv.CascadeClassifier();

        let response = await fetch("haarcascade_frontalface_default.xml");
        let data = await response.text();

        cv.FS_createDataFile("/", "haarcascade_frontalface_default.xml", data, true, false);

        faceCascade.load("haarcascade_frontalface_default.xml");
        console.log("Haarcascade loaded!");
    }

    await loadHaarCascade();

    // Start webcam
    async function startCamera() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: W, height: H },
                audio: false
            });
            video.srcObject = stream;
            video.playsInline = true;
            video.muted = true;
            await video.play();
            console.log("Webcam ready.");
            requestAnimationFrame(processFrame);
        } catch (err) {
            console.error("Camera error:", err);
        }
    }

    startCamera();

    // Stats
    let frameCount = 0;
    let lastTime = performance.now();

    function updateStats() {
        const now = performance.now();
        const elapsed = now - lastTime;
        if (elapsed >= 1000) {
            const fps = (frameCount * 1000 / elapsed).toFixed(1);
            fpsText.textContent = `FPS: ${fps}`;
            frameCount = 0;
            lastTime = now;

            if (performance.memory) {
                const used = (performance.memory.usedJSHeapSize / 1048576).toFixed(1);
                const limit = (performance.memory.jsHeapSizeLimit / 1048576).toFixed(0);
                memoryText.textContent = `${used}/${limit} MB`;
            }
        }
    }

   
    function applyHaarFaceBlur(src, dst) {
        cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

        // Run Haar only every N frames
        if (detectCounter % DETECT_INTERVAL === 0) {
            let faces = new cv.RectVector();
            let minSize = new cv.Size(30, 30);

            faceCascade.detectMultiScale(
                gray,
                faces,
                1.1,
                4,
                0,
                minSize
            );

            // Save faces to cache
            faceCache = [];
            for (let i = 0; i < faces.size(); i++) {
                faceCache.push(faces.get(i));
            }

            faces.delete();
        }

        detectCounter++;

        src.copyTo(dst);

        const intensity = parseInt(intensitySlider.value);
        let k = Math.floor(intensity * 0.6) + 5;
        if (k % 2 === 0) k++;

        for (let face of faceCache) {
            let rect = new cv.Rect(face.x, face.y, face.width, face.height);

            let roiSrc = dst.roi(rect);
            let roiBlur = new cv.Mat();
            let mask = new cv.Mat.zeros(face.height, face.width, cv.CV_32FC1);

            cv.GaussianBlur(roiSrc, roiBlur, new cv.Size(k, k), 0);

            // Create a FEATHERED elliptical mask
            let cx = face.width / 2;
            let cy = face.height / 2;
            let ax = (face.width / 2) * ellipseScale;
            let ay = (face.height / 2) * ellipseScale;

            for (let y = 0; y < face.height; y++) {
                for (let x = 0; x < face.width; x++) {

                    let dx = (x - cx) / ax;
                    let dy = (y - cy) / ay;
                    let d = dx * dx + dy * dy;

                    let alpha = 0;

                    if (d < 1.0) {
                        alpha = 1.0 - d;  
                    }

                    mask.floatPtr(y, x)[0] = alpha;
                }
            }

            let maskRGBA = new cv.Mat();
            cv.cvtColor(mask, maskRGBA, cv.COLOR_GRAY2RGBA);

            //  Alpha blending
            let blended = new cv.Mat();
            blended.create(roiSrc.rows, roiSrc.cols, cv.CV_8UC4);

            for (let y = 0; y < blended.rows; y++) {
                for (let x = 0; x < blended.cols; x++) {
                    let a = mask.floatAt(y, x);

                    for (let c = 0; c < 4; c++) {
                        let origVal = roiSrc.ucharPtr(y, x)[c];
                        let blurVal = roiBlur.ucharPtr(y, x)[c];
                        blended.ucharPtr(y, x)[c] = origVal * (1 - a) + blurVal * a;
                    }
                }
            }

            blended.copyTo(roiSrc);

            // cleanup
            roiSrc.delete();
            roiBlur.delete();
            mask.delete();
            maskRGBA.delete();
            blended.delete();
        }
    }


    // Filters
    function grayFilter(src, dst) {
        const intensity = parseInt(intensitySlider.value);
        cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
        gray.convertTo(gray, -1, 1.0 - intensity / 120.0, 0);
        cv.cvtColor(gray, dst, cv.COLOR_GRAY2RGBA);
    }

    function addNoise(src, dst) {
        const intensity = parseInt(intensitySlider.value);
        const amount = Math.min(255, intensity * 3);

        const noiseArr = new Uint8ClampedArray(src.rows * src.cols * 4);
        for (let i = 0; i < noiseArr.length; i++) {
            noiseArr[i] = Math.floor(Math.random() * amount);
        }

        const noiseMat = cv.matFromArray(src.rows, src.cols, cv.CV_8UC4, noiseArr);
        cv.addWeighted(src, 1.0, noiseMat, 0.5, 0, dst);
        noiseMat.delete();
    }

    function colorize(src, dst) {
        cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
        cv.applyColorMap(gray, color, cv.COLORMAP_JET);

        const intensity = parseInt(intensitySlider.value) / 100.0;
        cv.cvtColor(src, bgr, cv.COLOR_RGBA2BGR);
        cv.addWeighted(bgr, 1 - intensity, color, intensity, 0, color);
        cv.cvtColor(color, dst, cv.COLOR_BGR2RGBA);
    }

    function cartoon(src, dst) {
        const intensity = parseInt(intensitySlider.value);
        const k = Math.max(3, (Math.floor(intensity / 10) * 2 + 1));

        cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
        cv.medianBlur(gray, gray, k);
        cv.adaptiveThreshold(gray, edges, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 9, intensity / 10);

        cv.GaussianBlur(src, color, new cv.Size(k, k), 0);
        cv.cvtColor(edges, edges, cv.COLOR_GRAY2RGBA);
        cv.bitwise_and(color, edges, dst);
    }

    function posterize(src, dst) {
        const intensity = parseInt(intensitySlider.value);
        const levels = Math.max(2, Math.floor(intensity / 25) + 2);
        const step = 255.0 / (levels - 1);

        const lut = new cv.Mat(1, 256, cv.CV_8U);
        for (let i = 0; i < 256; i++) {
            lut.ucharPtr(0, i)[0] = Math.round(i / step) * step;
        }

        cv.LUT(src, lut, dst);
        lut.delete();
    }

    // Main loop
    function processFrame() {
        updateStats();
        frameCount++;

        try {
            ctx.drawImage(video, 0, 0, W, H);

            const imgData = ctx.getImageData(0, 0, W, H);
            src.data.set(imgData.data);

            const filter = filterSelect.value;

            switch (filter) {
                case "gray":
                    grayFilter(src, dst);
                    break;

                case "noisy":
                    addNoise(src, dst);
                    break;

                case "colorize":
                    colorize(src, dst);
                    break;

                case "cartoon":
                    cartoon(src, dst);
                    break;

                case "posterize":
                    posterize(src, dst);
                    break;

                case "faceblur_dnn":
                    applyHaarFaceBlur(src, dst);
                    break;

                default:
                    src.copyTo(dst);
            }

            cv.imshow(canvas, dst);

        } catch (err) {
            console.error("Frame error:", err);
        }
        requestAnimationFrame(processFrame);
    }
};