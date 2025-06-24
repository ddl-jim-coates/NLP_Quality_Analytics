pipeline {
    agent any
    
    environment {
        DOMINO_API_KEY = credentials('domino-api-key')
        DOMINO_URL = 'https://se-demo.domino.tech'
        DOMINO_PROJECT = 'jim_coates/NLP_Quality_Analytics'
    }
    
    stages {
        stage('Debug Info') {
            steps {
                script {
                    echo "🔍 Debug Information:"
                    echo "Branch Name: ${env.BRANCH_NAME ?: 'NOT SET'}"
                    echo "Git Branch: ${env.GIT_BRANCH ?: 'NOT SET'}"
                    echo "Build Number: ${BUILD_NUMBER}"
                    echo "Domino URL: ${DOMINO_URL}"
                    echo "Domino Project: ${DOMINO_PROJECT}"
                    echo "Full API URL: ${DOMINO_URL}/projects/${DOMINO_PROJECT}/jobs"
                }
            }
        }
        
        stage('Test Domino API Connection') {
            steps {
                script {
                    echo "🔍 Testing Domino API connection..."
                    
                    // First, let's try to get project info to verify the path is correct
                    try {
                        def projectResponse = httpRequest(
                            httpMode: 'GET',
                            url: "${DOMINO_URL}/projects/${DOMINO_PROJECT}",
                            customHeaders: [
                                [name: 'X-Domino-Api-Key', value: DOMINO_API_KEY, maskValue: true]
                            ],
                            validResponseCodes: '200:299,404'
                        )
                        
                        if (projectResponse.status == 200) {
                            echo "✅ Project found! Project API is working."
                        } else {
                            echo "❌ Project not found (404). Let's try some alternatives..."
                        }
                        
                    } catch (Exception e) {
                        echo "⚠️ Error testing project API: ${e.getMessage()}"
                    }
                    
                    // Let's also try without the project path to test basic API connectivity
                    try {
                        def baseResponse = httpRequest(
                            httpMode: 'GET',
                            url: "${DOMINO_URL}/projects",
                            customHeaders: [
                                [name: 'X-Domino-Api-Key', value: DOMINO_API_KEY, maskValue: true]
                            ],
                            validResponseCodes: '200:299,401,403,404'
                        )
                        
                        echo "Base API response status: ${baseResponse.status}"
                        
                    } catch (Exception e) {
                        echo "⚠️ Error testing base API: ${e.getMessage()}"
                    }
                }
            }
        }
        
        stage('Try Alternative Project Paths') {
            steps {
                script {
                    echo "🔍 Testing alternative project paths..."
                    
                    def alternativePaths = [
                        'jim_coates/NLP_Quality_Analytics',
                        'jim-coates/NLP_Quality_Analytics', 
                        'jim_coates/nlp_quality_analytics',
                        'jim_coates/nlp-quality-analytics'
                    ]
                    
                    alternativePaths.each { projectPath ->
                        try {
                            echo "Testing path: ${projectPath}"
                            def response = httpRequest(
                                httpMode: 'GET',
                                url: "${DOMINO_URL}/projects/${projectPath}",
                                customHeaders: [
                                    [name: 'X-Domino-Api-Key', value: DOMINO_API_KEY, maskValue: true]
                                ],
                                validResponseCodes: '200:299,404'
                            )
                            
                            if (response.status == 200) {
                                echo "✅ FOUND! Correct project path: ${projectPath}"
                                env.CORRECT_PROJECT_PATH = projectPath
                                return true // Break out of the loop
                            } else {
                                echo "❌ ${projectPath} - Not found"
                            }
                            
                        } catch (Exception e) {
                            echo "❌ ${projectPath} - Error: ${e.getMessage()}"
                        }
                    }
                }
            }
        }
        
        stage('Trigger Domino Training Job') {
            when {
                expression { env.CORRECT_PROJECT_PATH != null }
            }
            steps {
                script {
                    echo "🚀 Starting Domino job with correct path: ${env.CORRECT_PROJECT_PATH}"
                    
                    def jobPayload = [
                        command: ["python", "experiments/mlflow_tracking.py"],
                        isDirect: false,
                        title: "Jenkins Triggered Training - Build ${BUILD_NUMBER}",
                        tier: "Free",
                        publishApiEndpoint: false
                    ]
                    
                    def response = httpRequest(
                        httpMode: 'POST',
                        url: "${DOMINO_URL}/projects/${env.CORRECT_PROJECT_PATH}/jobs",
                        customHeaders: [
                            [name: 'X-Domino-Api-Key', value: DOMINO_API_KEY, maskValue: true],
                            [name: 'Content-Type', value: 'application/json']
                        ],
                        requestBody: groovy.json.JsonOutput.toJson(jobPayload),
                        validResponseCodes: '200:299'
                    )
                    
                    def jobInfo = readJSON text: response.content
                    env.DOMINO_JOB_ID = jobInfo.id
                    echo "✅ Started Domino job: ${jobInfo.id}"
                }
            }
        }
        
        stage('Fallback - Manual Job Start') {
            when {
                expression { env.CORRECT_PROJECT_PATH == null }
            }
            steps {
                script {
                    echo "❌ Could not find correct project path automatically."
                    echo "🔧 Manual steps to fix:"
                    echo "1. Go to your Domino project: ${DOMINO_URL}/projects"
                    echo "2. Find your project 'NLP_Quality_Analytics'"
                    echo "3. Look at the URL - it should be something like:"
                    echo "   ${DOMINO_URL}/workspace/username/projectname"
                    echo "4. Update DOMINO_PROJECT variable in Jenkinsfile with correct path"
                    echo ""
                    echo "Common patterns:"
                    echo "- jim-coates/nlp-quality-analytics (kebab-case)"
                    echo "- jim_coates/nlp_quality_analytics (snake_case)"
                    echo "- JimCoates/NLPQualityAnalytics (no spaces/underscores)"
                    
                    currentBuild.result = 'UNSTABLE'
                }
            }
        }
    }
    
    post {
        always {
            echo "🏁 Pipeline completed"
        }
        success {
            echo "✅ Successfully started Domino training job!"
        }
        failure {
            echo "❌ Pipeline failed"
        }
        unstable {
            echo "⚠️ Pipeline completed with issues - check project path"
        }
    }
}