plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("dev.flutter.flutter-gradle-plugin")
}

android {
    namespace = "com.example.parking_empty_mlkit"
    compileSdk = 35          // 옵션 B → 35
    ndkVersion = flutter.ndkVersion

    defaultConfig {
        applicationId = "com.example.parking_empty_mlkit"
        minSdk = 24
        targetSdk = 35       // 옵션 B → 35
        versionCode = flutter.versionCode
        versionName = flutter.versionName
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }

    buildTypes {
        release { signingConfig = signingConfigs.getByName("debug") }
    }
}

flutter { source = "../.." }

dependencies {
    // 추가 라이브러리 필요 시 여기에 (보통 비움)
}
