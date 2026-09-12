// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "XLD-L2D",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "XLDKit", targets: ["XLDKit"]),
        .executable(name: "xld", targets: ["XLDCLI"]),
        .executable(name: "XLDApp", targets: ["XLDApp"]),
    ],
    targets: [
        .target(name: "XLDKit", path: "Sources/XLDKit"),
        .executableTarget(name: "XLDCLI", dependencies: ["XLDKit"], path: "Sources/XLDCLI"),
        .executableTarget(
            name: "XLDApp",
            dependencies: ["XLDKit"],
            path: "Sources/XLDApp"
        ),
        .testTarget(name: "XLDKitTests", dependencies: ["XLDKit"], path: "Tests/XLDKitTests"),
    ]
)
