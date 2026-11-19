#include "app/classifier_client.h"

#include <HTTPClient.h>
#include <WiFi.h>
#include <WiFiClient.h>
#include <esp_heap_caps.h>
#include <cstring>

namespace classifier_client {

bool upload_jpeg(const char* url, const uint8_t* jpeg_data, size_t jpeg_len,
                 String& json_result, int& http_status, uint32_t timeout_ms) {
    json_result = "";
    http_status = -1;
    if (!url || !jpeg_data || jpeg_len == 0 || WiFi.status() != WL_CONNECTED)
        return false;

    // Simple first implementation. Copies JPEG into a multipart body in PSRAM.
    // YOLO11n may leave little PSRAM, so measure this allocation on real hardware.
    const char* boundary = "SmartSortFrameBoundary";
    String head = String("--") + boundary + "\r\n" +
                  "Content-Disposition: form-data; name=\"file\"; filename=\"item.jpg\"\r\n" +
                  "Content-Type: image/jpeg\r\n\r\n";
    String tail = String("\r\n--") + boundary + "--\r\n";
    const size_t size = head.length() + jpeg_len + tail.length();
    if (size < jpeg_len) return false; // overflow guard
    uint8_t* body = static_cast<uint8_t*>(
        heap_caps_malloc(size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    if (!body) {
        Serial.println("[API] Could not allocate multipart body in PSRAM");
        return false;
    }
    std::memcpy(body, head.c_str(), head.length());
    std::memcpy(body + head.length(), jpeg_data, jpeg_len);
    std::memcpy(body + head.length() + jpeg_len, tail.c_str(), tail.length());

    WiFiClient client;
    HTTPClient http;
    bool ok = false;
    if (http.begin(client, url)) {
        http.setConnectTimeout(timeout_ms);
        http.setTimeout(timeout_ms);
        http.addHeader("Content-Type", String("multipart/form-data; boundary=") + boundary);
        http_status = http.POST(body, size);
        if (http_status > 0) json_result = http.getString();
        ok = http_status == HTTP_CODE_OK;
        http.end();
    }
    heap_caps_free(body);
    return ok;
}

} // namespace classifier_client
