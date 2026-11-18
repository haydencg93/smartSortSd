
#include "app/demo_flow.h"
#include "ui/ui.h"
#include "sensors/camera.h"

#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <WiFiClient.h>
#include <cJSON.h>

#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <freertos/queue.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Classifier configuration
// ---------------------------------------------------------------------------

static const char* CLASSIFIER_URL =
    "http://10.67.162.143:8000/classify";

static const char* MULTIPART_BOUNDARY =
    "----SmartSortBoundary7MA4YWxkTrZu0gW";

// ---------------------------------------------------------------------------
// Flow state
// ---------------------------------------------------------------------------

static lv_timer_t* pending_flow_timer = nullptr;
static lv_timer_t* result_poll_timer = nullptr;

static QueueHandle_t classifier_result_queue = nullptr;

static volatile bool classifier_busy = false;
static volatile bool flow_active = false;

static uint32_t flow_generation = 0;

// ---------------------------------------------------------------------------
// Data passed between the main loop and classifier task
// ---------------------------------------------------------------------------

struct ClassificationResult {
    uint32_t generation;

    bool success;

    char classification[32];
    char material[48];

    float confidence;
    float latency_ms;
};

struct ClassificationRequest {
    uint32_t generation;

    uint8_t* jpeg_data;
    size_t jpeg_size;
};

// ---------------------------------------------------------------------------
// Timer helper
// ---------------------------------------------------------------------------

static void after(uint32_t ms, lv_timer_cb_t fn) {
    if (pending_flow_timer != nullptr) {
        lv_timer_delete(pending_flow_timer);
    }

    pending_flow_timer = lv_timer_create(fn, ms, nullptr);
    lv_timer_set_repeat_count(pending_flow_timer, 1);
}

// ---------------------------------------------------------------------------
// JSON parsing
// ---------------------------------------------------------------------------

static bool parse_classifier_response(
    const String& response,
    ClassificationResult& result
) {
    cJSON* root = cJSON_Parse(response.c_str());

    if (root == nullptr) {
        Serial.println("[API] ERROR: Invalid JSON response.");
        return false;
    }

    cJSON* classification =
        cJSON_GetObjectItemCaseSensitive(root, "classification");

    cJSON* material =
        cJSON_GetObjectItemCaseSensitive(root, "material");

    cJSON* confidence =
        cJSON_GetObjectItemCaseSensitive(root, "confidence");

    cJSON* latency =
        cJSON_GetObjectItemCaseSensitive(root, "latency_ms");

    bool valid =
        cJSON_IsString(classification) &&
        cJSON_IsString(material) &&
        cJSON_IsNumber(confidence) &&
        cJSON_IsNumber(latency);

    if (valid) {
        snprintf(
            result.classification,
            sizeof(result.classification),
            "%s",
            classification->valuestring
        );

        snprintf(
            result.material,
            sizeof(result.material),
            "%s",
            material->valuestring
        );

        result.confidence = static_cast<float>(confidence->valuedouble);
        result.latency_ms = static_cast<float>(latency->valuedouble);
    }

    cJSON_Delete(root);

    return valid;
}

// ---------------------------------------------------------------------------
// Multipart HTTP upload
// ---------------------------------------------------------------------------

static bool classify_image(
    uint8_t* jpeg_data,
    size_t jpeg_size,
    ClassificationResult& result
) {
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("[API] ERROR: WiFi is disconnected.");
        return false;
    }

    WiFiClient client;
    HTTPClient http;

    http.setConnectTimeout(5000);
    http.setTimeout(15000);

    Serial.println("[API] Connecting to classifier...");

    if (!http.begin(client, CLASSIFIER_URL)) {
        Serial.println("[API] ERROR: Could not initialize HTTP.");
        return false;
    }

    String content_type =
        String("multipart/form-data; boundary=") + MULTIPART_BOUNDARY;

    http.addHeader("Content-Type", content_type);
    http.addHeader("Connection", "close");

    char multipart_header[256];

    int header_length = snprintf(
        multipart_header,
        sizeof(multipart_header),
        "--%s\r\n"
        "Content-Disposition: form-data; name=\"file\"; filename=\"capture.jpg\"\r\n"
        "Content-Type: image/jpeg\r\n\r\n",
        MULTIPART_BOUNDARY
    );

    char multipart_footer[96];

    int footer_length = snprintf(
        multipart_footer,
        sizeof(multipart_footer),
        "\r\n--%s--\r\n",
        MULTIPART_BOUNDARY
    );

    if (header_length <= 0 ||
        footer_length <= 0 ||
        static_cast<size_t>(header_length) >= sizeof(multipart_header) ||
        static_cast<size_t>(footer_length) >= sizeof(multipart_footer)) {

        Serial.println("[API] ERROR: Multipart formatting failed.");
        http.end();
        return false;
    }

    size_t total_size =
        static_cast<size_t>(header_length) +
        jpeg_size +
        static_cast<size_t>(footer_length);

    uint8_t* body = static_cast<uint8_t*>(malloc(total_size));

    if (body == nullptr) {
        Serial.println("[API] ERROR: Could not allocate upload buffer.");
        http.end();
        return false;
    }

    memcpy(body, multipart_header, header_length);

    memcpy(
        body + header_length,
        jpeg_data,
        jpeg_size
    );

    memcpy(
        body + header_length + jpeg_size,
        multipart_footer,
        footer_length
    );

    Serial.printf(
        "[API] Uploading JPEG: %u bytes\n",
        static_cast<unsigned int>(jpeg_size)
    );

    int status_code = http.POST(body, total_size);

    free(body);

    bool success = false;

    if (status_code == HTTP_CODE_OK) {
        String response = http.getString();

        Serial.println("[API] Classifier response:");
        Serial.println(response);

        success = parse_classifier_response(response, result);
    } else {
        Serial.printf(
            "[API] HTTP request failed. Status: %d\n",
            status_code
        );

        if (status_code < 0) {
            Serial.println(http.errorToString(status_code));
        }
    }

    http.end();

    return success;
}

// ---------------------------------------------------------------------------
// Background classifier task
// ---------------------------------------------------------------------------

static void classifier_task(void* parameter) {
    ClassificationRequest* request =
        static_cast<ClassificationRequest*>(parameter);

    ClassificationResult result = {};

    result.generation = request->generation;

    Serial.println("[API] Classification task started.");

    result.success = classify_image(
        request->jpeg_data,
        request->jpeg_size,
        result
    );

    free(request->jpeg_data);
    free(request);

    Serial.printf(
        "[API] Classification task finished. Success: %s\n",
        result.success ? "yes" : "no"
    );

    if (result.success) {
        Serial.printf(
            "[API] Classification: %s | Material: %s | Confidence: %.4f | Latency: %.1f ms\n",
            result.classification,
            result.material,
            result.confidence,
            result.latency_ms
        );
    }

    // Send the result to the LVGL/main-loop side.
    if (classifier_result_queue != nullptr) {
        xQueueOverwrite(classifier_result_queue, &result);
    }

    classifier_busy = false;

    vTaskDelete(nullptr);
}

// ---------------------------------------------------------------------------
// Start background classification
// ---------------------------------------------------------------------------

static void start_classifier(uint8_t* jpeg_data, size_t jpeg_size) {
    if (classifier_busy) {
        Serial.println("[API] ERROR: A classification is already running.");

        free(jpeg_data);
        ui::show_manual_review();
        return;
    }

    ClassificationRequest* request =
        static_cast<ClassificationRequest*>(
            malloc(sizeof(ClassificationRequest))
        );

    if (request == nullptr) {
        Serial.println("[API] ERROR: Could not allocate request.");

        free(jpeg_data);
        ui::show_manual_review();
        return;
    }

    request->generation = flow_generation;
    request->jpeg_data = jpeg_data;
    request->jpeg_size = jpeg_size;

    classifier_busy = true;

    BaseType_t task_created = xTaskCreate(
        classifier_task,
        "ClassifierTask",
        12288,
        request,
        1,
        nullptr
    );

    if (task_created != pdPASS) {
        Serial.println("[API] ERROR: Could not create classifier task.");

        classifier_busy = false;

        free(request->jpeg_data);
        free(request);

        ui::show_manual_review();
    }
}

// ---------------------------------------------------------------------------
// Main-loop result handling
// ---------------------------------------------------------------------------

static void poll_classifier_result(lv_timer_t*) {
    if (classifier_result_queue == nullptr) {
        return;
    }

    ClassificationResult result = {};

    if (xQueueReceive(
            classifier_result_queue,
            &result,
            0
        ) != pdTRUE) {
        return;
    }

    // Ignore results belonging to an earlier or cancelled scan.
    if (!flow_active || result.generation != flow_generation) {
        Serial.println("[API] Ignoring an outdated classification result.");
        return;
    }

    flow_active = false;

    if (!result.success) {
        Serial.println("[API] Classification failed. Requesting manual review.");
        ui::show_manual_review();
        return;
    }

    if (strcmp(result.classification, "recycling") == 0) {
        ui::show_result(
            ui::Bin::Recycle,
            result.material,
            result.confidence,
            result.latency_ms
        );
    }
    else if (strcmp(result.classification, "landfill") == 0) {
        ui::show_result(
            ui::Bin::Trash,
            result.material,
            result.confidence, 
            result.latency_ms
        );
    }
    else if (strcmp(result.classification, "special_disposal") == 0) {
        ui::show_special_disposal(result.confidence, result.latency_ms);
    }
    else if (strcmp(result.classification, "manual_review") == 0) {
        ui::show_manual_review(result.confidence, result.latency_ms);
    }
    else {
        Serial.println("[API] ERROR: Unknown classification.");
        ui::show_manual_review();
    }
}

// ---------------------------------------------------------------------------
// Scan flow
// ---------------------------------------------------------------------------

static void step_analyze(lv_timer_t* timer) {
    if (pending_flow_timer == timer) {
        pending_flow_timer = nullptr;
    }

    if (!flow_active) {
        return;
    }

    Serial.println("[Camera] Capturing final image...");

    // Capture a JPEG before show_analyzing() shuts down the camera.
    uint8_t* jpeg_data = nullptr;
    size_t jpeg_size = 0;

    if (!camera::capture_jpeg(&jpeg_data, jpeg_size)) {
        Serial.println("[Camera] ERROR: JPEG capture failed.");
        ui::show_manual_review();
        flow_active = false;
        return;
    }

    Serial.printf(
        "[Camera] JPEG ready: %u bytes\n",
        static_cast<unsigned int>(jpeg_size)
    );

    // This stops the camera and displays the analyzing screen.
    ui::show_analyzing();

    // The actual HTTP request runs in a separate task.
    start_classifier(jpeg_data, jpeg_size);
}

static void on_idle_tapped() {
    if (classifier_busy) {
        Serial.println("[API] Previous classification is still running.");
        return;
    }

    flow_generation++;
    flow_active = true;

    Serial.println("[Flow] Starting new scan.");

    ui::show_scan_prompt();

    after(5000, step_analyze);
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

void demo_flow_start_scan() {
    on_idle_tapped();
}

void demo_flow_cancel() {
    flow_generation++;
    flow_active = false;

    if (pending_flow_timer != nullptr) {
        lv_timer_delete(pending_flow_timer);
        pending_flow_timer = nullptr;
    }

    Serial.println("[Flow] Scan cancelled.");
}

static void on_feedback(const ui::Feedback& f) {
    printf(
        "[feedback] item=\"%s\" shown=%s correct=%s reason=%d\n",
        f.item,
        f.shown == ui::Bin::Recycle ? "recycle" : "trash",
        f.correct == ui::Bin::Recycle ? "recycle" : "trash",
        static_cast<int>(f.reason)
    );
}

void demo_flow_init() {
    classifier_result_queue = xQueueCreate(
        1,
        sizeof(ClassificationResult)
    );

    if (classifier_result_queue == nullptr) {
        Serial.println("[API] ERROR: Could not create result queue.");
    }

    // This timer runs on the LVGL thread and safely applies results.
    result_poll_timer = lv_timer_create(
        poll_classifier_result,
        100,
        nullptr
    );

    ui::Callbacks cb;

    cb.on_idle_tapped = on_idle_tapped;
    cb.on_feedback = on_feedback;

    ui::init(cb);

    Serial.println("[Flow] Real classifier integration initialized.");
}
