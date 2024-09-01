var pixels;

let thresh = 25;

function processImg() {
    var uploader = document.querySelector("input[type=file]").files[0];
    var canvas = document.getElementById("cnvs");
    var ctx = canvas.getContext("2d");
    var reader = new FileReader();
    var img = new Image();

    reader.addEventListener("load", () => {
        img.src = reader.result;
    });

    if (uploader) {
        reader.readAsDataURL(uploader);
    }

    img.onload = () => {
        let output = document.querySelector("#result");
        output.value = "";
        document.getElementById("filename").innerText = uploader.name;
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0, img.width, img.height);
        console.log("Image drawing complete.");
        pixels = ctx.getImageData(0, 0, img.width, img.height);
        let lines = [];
        detectLines(lines);
        console.log("DONE! Created " + lines.length + " lines.");
        output.value = "";
        lines.forEach((line) => {
            output.value += getLineEquation(line);
        });

        output.select();
    };
}

function detectLines(lines) {
    for (let y = 0; y < pixels.height; y++) {
        let lineStart = null;
        for (let x = 0; x < pixels.width; x++) {
            if (isEdge(x, y)) {
                if (lineStart === null) {
                    lineStart = getX(x);
                }
            } else {
                if (lineStart !== null) {
                    lines.push({
                        dir: 1,
                        offset: getY(y),
                        start: lineStart,
                        end: getX(x - 1),
                    });
                    lineStart = null;
                }
            }
        }
        if (lineStart !== null) {
            lines.push({
                dir: 1,
                offset: getY(y),
                start: lineStart,
                end: getX(pixels.width - 1),
            });
        }
    }

    for (let x = 0; x < pixels.width; x++) {
        let lineStart = null;
        for (let y = 0; y < pixels.height; y++) {
            if (isEdge(x, y)) {
                if (lineStart === null) {
                    lineStart = getY(y);
                }
            } else {
                if (lineStart !== null) {
                    lines.push({
                        dir: 0,
                        offset: getX(x),
                        start: lineStart,
                        end: getY(y - 1),
                    });
                    lineStart = null;
                }
            }
        }
        if (lineStart !== null) {
            lines.push({
                dir: 0,
                offset: getX(x),
                start: lineStart,
                end: getY(pixels.height - 1),
            });
        }
    }

    lines = mergeLines(lines);
}

function isEdge(x, y) {
    let gradient = 0;
    for (let k = 0; k < 4; k++) {
        let diffX = Math.abs(getPixel(x, y, k) - getPixel(x + 1, y, k));
        let diffY = Math.abs(getPixel(x, y, k) - getPixel(x, y + 1, k));
        gradient += Math.max(diffX, diffY);
    }
    return gradient >= thresh;
}

function mergeLines(lines) {
    let merged = [];
    let current = null;
    for (let i = 0; i < lines.length; i++) {
        if (!current) {
            current = lines[i];
        } else if (
            current.dir == lines[i].dir &&
            Math.abs(current.end - lines[i].start) < 5 &&
            Math.abs(current.offset - lines[i].offset) < 5
        ) {
            current.end = lines[i].end;
        } else {
            merged.push(current);
            current = lines[i];
        }
    }
    if (current) merged.push(current);
    return merged;
}

async function copyToClipboard() {
    var copyText = document.querySelector("#result");
    await navigator.clipboard.writeText(copyText.value);

    var copyStatus = document.getElementById("copy-status");
    copyStatus.innerText = "Copied!";

    setTimeout(() => {
        copyStatus.innerText = "";
    }, 1000);
}

function getLineEquation(line) {
    if (line.dir == 0) {
        return (
            "x = " +
            line.offset +
            "\\left\\{" +
            line.start +
            "\\le y \\le" +
            line.end +
            "\\right\\}\n"
        );
    } else {
        return (
            "y = " +
            line.offset +
            "\\left\\{" +
            line.start +
            "\\le x \\le" +
            line.end +
            "\\right\\}\n"
        );
    }
}

function getPixel(i, j, c) {
    return pixels.data[j * 4 * pixels.width + 4 * i + c];
}

function getX(x) {
    return x - pixels.width / 2;
}
function getY(y) {
    return -y + pixels.height / 2;
}
