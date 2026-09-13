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
            path: "Sources/XLDApp",
            exclude: ["Resources/Info.plist"],
            linkerSettings: [
                .unsafeFlags([
                    "-Xlinker", "-sectcreate",
                    "-Xlinker", "__TEXT",
                    "-Xlinker", "__info_plist",
                    "-Xlinker", "Sources/XLDApp/Resources/Info.plist"
                ])
            ]
        ),
        .testTarget(name: "XLDKitTests", dependencies: ["XLDKit"], path: "Tests/XLDKitTests"),
    ]
)
