plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("dev.flutter.flutter-gradle-plugin")
}

android {
    namespace = "com.example.parking_empty_mlkit"
    compileSdk = 35

    // flutter.gradle이 주입하는 값 사용
    ndkVersion = flutter.ndkVersion

    defaultConfig {
        applicationId = "com.example.parking_empty_mlkit"
        minSdk = 24
        targetSdk = 35
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
    // 필요 시만 추가
}
