import org.gradle.api.initialization.resolve.RepositoriesMode
import java.io.File

pluginManagement {
    // Flutter SDK 경로(local.properties → flutter.sdk) 읽기
    val flutterSdkPath = run {
        val f = File("local.properties")
        check(f.exists()) { "local.properties not found. It must contain flutter.sdk=/path/to/flutter" }
        val props = java.util.Properties()
        f.inputStream().use { input -> props.load(input) }   // ← 리시버 혼동 방지
        val p = props.getProperty("flutter.sdk")
        check(p != null) { "flutter.sdk not set in local.properties" }
        p!!
    }

    includeBuild("$flutterSdkPath/packages/flutter_tools/gradle")

    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
        maven(url = uri("https://storage.googleapis.com/download.flutter.io"))
    }

    // ⚠ 버전은 settings 에서만 고정. app 쪽에는 버전 쓰지 않음.
    plugins {
        id("com.android.application") version "8.12.0"       // 옵션 B
        id("org.jetbrains.kotlin.android") version "2.1.10" // 1.9.25 대신 2.0.21 (AGP 8.9.1과 호환)
        id("dev.flutter.flutter-gradle-plugin") version "1.0.0"
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.PREFER_SETTINGS)
    repositories {
        google()
        mavenCentral()
        maven(url = uri("https://storage.googleapis.com/download.flutter.io"))
    }
}

rootProject.name = "parking_empty_mlkit"
include(":app")
